import re
import io
import os
import httpx
import fitz
import hashlib
import uuid
import docx
import openpyxl
from pathlib import Path
from typing import List
from backend.shared.logger import setup_logger
from backend.shared.config import get_settings


settings = get_settings()
logger = setup_logger("chunker")


# ── Types supportés ────────────────────────
SUPPORTED_TEXT_TYPES = {
    "application/pdf",
    #"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    #"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #"text/plain",
    #"text/html",
    #"text/markdown",
}

class RecursiveCharacterTextSplitter:
    """
    Réimplémentation minimale de RecursiveCharacterTextSplitter.
    Même comportement, zéro dépendance.
    """

    def __init__(
        self,
        chunk_size:     int       = 900,
        chunk_overlap:  int       = 150,
        separators:     list[str] = None,
        length_function=len,
    ):
        self.chunk_size      = settings.chunk_size
        self.chunk_overlap   = settings.chunk_overlap
        self.length_function = length_function
        self.separators      = separators or [
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ]

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []

        # Trouver le bon séparateur
        separator   = ""
        new_seps    = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_seps  = separators[i + 1:]
                break

        # Découper avec le séparateur trouvé
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Fusionner les petits morceaux
        good_splits  = []
        current      = []
        current_len  = 0

        for split in splits:
            split_len = self.length_function(split)

            if current_len + split_len + (
                len(separator) if current else 0
            ) > self.chunk_size:
                if current:
                    merged = separator.join(current).strip()
                    if merged:
                        good_splits.append(merged)

                    # Overlap : garder la fin
                    while current and current_len > self.chunk_overlap:
                        removed     = current.pop(0)
                        current_len -= self.length_function(removed) + len(separator)

                current.append(split)
                current_len = split_len
            else:
                current.append(split)
                current_len += split_len + (
                    len(separator) if len(current) > 1 else 0
                )

        # Dernier morceau
        if current:
            merged = separator.join(current).strip()
            if merged:
                good_splits.append(merged)

        # Re-découper les morceaux encore trop grands
        final = []
        for chunk in good_splits:
            if self.length_function(chunk) > self.chunk_size and new_seps:
                sub = self._split_text(chunk, new_seps)
                final.extend(sub)
            else:
                if chunk.strip():
                    final.append(chunk)

        return final


def _is_valid_chunk(text: str) -> bool:
    """
    Vérifie qu'un chunk a suffisamment de contenu
    pour valoir la peine d'être indexé.
    """
    if not text or not text.strip():
        return False

    # Trop court en caractères
    if len(text.strip()) < settings.min_chunk_chars:
        return False

    # Trop peu de mots
    words = [w for w in text.split() if len(w) > 1]
    if len(words) < settings.min_chunk_words:
        return False

    return True


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Supprime les chunks dupliqués basé sur le texte.
    """
    seen   = set()
    result = []

    for chunk in chunks:
        text_key = chunk.get("text", "").strip()

        if text_key not in seen:
            seen.add(text_key)
            result.append(chunk)
        else:
            logger.debug(
                f"_deduplicate_chunks | doublon supprimé | "
                f"'{text_key[:50]}'"
            )

    return result

def compute_chunk_hash(chunk_text: str, metadata: dict) -> str:
    """
    UUID v5 stable et unique par chunk.
    Utilise le texte complet (pas juste 100 chars) pour éviter les collisions.
    """
    source = metadata.get("source", "")

    if source in ("issue", "issue_attachment"):
        identity = "|".join([
            "issue",
            str(metadata.get("issue_id",      "")),
            str(metadata.get("attachment_id", "")),
            str(metadata.get("component",     "")),
            str(metadata.get("chunk_index",   0)),
            chunk_text.strip(),   
        ])

    elif source in ("wiki", "wiki_attachment"):
        identity = "|".join([
            "wiki",
            str(metadata.get("project_id",    "")),
            str(metadata.get("title",         "")),
            str(metadata.get("wiki_title",    "")),
            str(metadata.get("attachment_id", "")),
            str(metadata.get("chunk_index",   0)),
            chunk_text.strip(),
        ])

    elif source == "project":
        identity = "|".join([
            "project",
            str(metadata.get("project_id",  "")),
            str(metadata.get("chunk_index", 0)),
            chunk_text.strip(),
        ])

    else:
        identity = "|".join([
            str(source),
            str(metadata.get("project_id",  "")),
            str(metadata.get("chunk_index", 0)),
            chunk_text.strip(),
        ])

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, identity))

class MarkdownHeaderTextSplitter:
    """
    Réimplémentation minimale de MarkdownHeaderTextSplitter.
    Découpe le texte selon les headers Markdown.
    Retourne une liste de dicts {"page_content": ..., "metadata": {...}}
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple] = None,
        strip_headers:       bool        = False,
    ):
        self.headers_to_split_on = headers_to_split_on or [
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ]
        self.strip_headers = strip_headers

        # Trier par longueur décroissante (### avant ##  avant #)
        self.headers_to_split_on = sorted(
            self.headers_to_split_on,
            key=    lambda x: len(x[0]),
            reverse=True,
        )

    def split_text(self, text: str) -> list:
        """
        Retourne une liste d'objets avec .page_content et .metadata
        """
        if not text.strip():
            return []

        lines    = text.split("\n")
        chunks   = []
        current_content = []
        current_metadata = {}

        for line in lines:
            header_found = False

            for sep, name in self.headers_to_split_on:
                # Vérifier si la ligne est un header
                pattern = f"^{re.escape(sep)}\\s+(.+)$"
                match   = re.match(pattern, line.strip())

                if match:
                    # Sauvegarder le chunk précédent
                    content = "\n".join(current_content).strip()
                    if content:
                        chunks.append(
                            _Document(
                                page_content=content,
                                metadata=    dict(current_metadata),
                            )
                        )

                    # Nouveau chunk
                    header_text = match.group(1).strip()
                    current_content  = [] if self.strip_headers else [line]
                    current_metadata = dict(current_metadata)

                    # Mettre à jour la metadata pour ce niveau
                    current_metadata[name] = header_text

                    # Effacer les niveaux inférieurs
                    level = len(sep)
                    for s, n in self.headers_to_split_on:
                        if len(s) > level:
                            current_metadata.pop(n, None)

                    header_found = True
                    break

            if not header_found:
                current_content.append(line)

        # Dernier chunk
        content = "\n".join(current_content).strip()
        if content:
            chunks.append(
                _Document(
                    page_content=content,
                    metadata=    dict(current_metadata),
                )
            )

        return chunks


class _Document:
    """
    Objet document minimal compatible avec LangChain Document.
    """
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata     = metadata
# ─────────────────────────────────────────────
#   SSL
# ─────────────────────────────────────────────

def get_ssl_verify() -> str | bool:
    """
    Retourne la configuration SSL à utiliser pour httpx.

    Retourne :
    - str  → chemin vers le certificat CA si trouvé
    - False → désactiver la vérification SSL (fallback)

    """
    ssl_cert = settings.redmine_ssl_cert_path

    if ssl_cert:
        cert_path = Path(ssl_cert)

        if cert_path.exists():
            logger.info(f"SSL | certificat utilisé : {cert_path}")
            return str(cert_path)       
        else:
            logger.warning(
                f"SSL | certificat introuvable : {cert_path} "
                f"| fallback verify=False"
            )
            return False                  
    logger.warning(
        "SSL | certificat redmine introuvable | fallback verify=False"
    )
    return False


# ─────────────────────────────────────────────
#   NETTOYAGE DU TEXTE
# ─────────────────────────────────────────────
def clean_textile(text: str) -> str:
    if not text:
        return ""

    # ── 1. Blocs <pre><code> ──────────────────────────────────────
    def replace_code_block(match):
        content = match.group(1).strip()
        lines   = [l for l in content.split('\n') if l.strip()]
        if not lines:
            return ""
        if len(content) > 100:
            preview = '\n'.join(lines[:3])
            return (
                f"\n[bloc de code - {len(lines)} lignes]\n"
                f"{preview}\n...\n"
            )
        return f"\n{content}\n"

    text = re.sub(
        r'<pre>\s*<code[^>]*>(.*?)</code>\s*</pre>',
        replace_code_block,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ── 2. Blocs <pre> ────────────────────────────────────────────
    def replace_pre_block(match):
        content = match.group(1).strip()
        lines   = [l for l in content.split('\n') if l.strip()]
        if not lines:
            return ""
        if len(content) > 300:
            preview = '\n'.join(lines[:3])
            return (
                f"\n[bloc préformaté - {len(lines)} lignes]\n"
                f"{preview}\n...\n"
            )
        return f"\n{content}\n"

    text = re.sub(
        r'<pre>(.*?)</pre>',
        replace_pre_block,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ── 3. Notextile ──────────────────────────────────────────────
    text = re.sub(r'<notextile>.*?</notextile>', '', text, flags=re.DOTALL)
    text = re.sub(r'<notextile>|</notextile>',   '', text)

    # ── 4. Macros ─────────────────────────────────────────────────
    text = re.sub(r'\{\{[^}]*\}\}', '', text, flags=re.DOTALL)

    # ── 5. Headings h1..h6 ────────────────────────────────────────
    for level in range(1, 7):
        text = re.sub(
            rf'^h{level}\.\s*(.+)$',
            r'\1',
            text,
            flags=re.MULTILINE,
        )

    # ── 6. Paragraphes p=. etc. ───────────────────────────────────
    text = re.sub(
        r'^p[\(\)><={}\d]*\.\s*(.*)$',
        r'\1',
        text,
        flags=re.MULTILINE,
    )

    # ── 7. Blockquotes bq. ────────────────────────────────────────
    text = re.sub(r'^bq\.\s*(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^>{1,3}\s*',    '',    text, flags=re.MULTILINE)

    # ── 8. CSS %{color:red}texte% ────────────────────────────────
    text = re.sub(r'%\{[^}]*\}([^%]*)%', r'\1', text)

    # ── 9. Liens "texte":url ──────────────────────────────────────
    text = re.sub(r'"([^"]+)":\S+', r'\1', text)

    # ── 10. Liens Wiki [[Page|texte]] ────────────────────────────
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]',             r'\1', text)

    # ── 11. Images ───────────────────────────────────────────────
    text = re.sub(r'!\{[^}]*\}[^!]+!',              '', text)
    text = re.sub(r'![<>]?[^!(]+\([^)]*\)!(?::\S+)?', '', text)
    text = re.sub(r'![<>]?[^!\s][^!]*!(?::\S+)?',   '', text)

    # ── 12. Gras *bold* ──────────────────────────────────────────
    # ⚠️ Seulement si * entoure du texte sans underscore/slash
    text = re.sub(r'(?<!\w)\*([^*\n]+)\*(?!\w)', r'\1', text)

    # ── 13. Italique _italic_ ────────────────────────────────────
    # ⚠️ FIX CRITIQUE : pas les underscores dans les noms de variables !
    # _italic_ = underscore entouré de non-word chars
    text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'\1', text)

    # ── 14. Souligné +underline+ ─────────────────────────────────
    text = re.sub(r'(?<!\w)\+([^+\n]+)\+(?!\w)', r'\1', text)

    # ── 15. Barré -strike- ───────────────────────────────────────
    text = re.sub(r'(?<!\w)-([^-\n]{2,})-(?!\w)', r'\1', text)

    # ── 16. Exposant ^superscript^ ───────────────────────────────
    text = re.sub(r'\^([^\^\n]+)\^', r'\1', text)

    # ── 17. Indice ~subscript~ ───────────────────────────────────
    text = re.sub(r'(?<!\w)~([^~\n]+)~(?!\w)', r'\1', text)

    # ── 18. Monospace @inline@ ───────────────────────────────────
    # ⚠️ Seulement si @ entoure du texte court (pas email)
    text = re.sub(r'(?<!\w)@([^@\n]{1,50})@(?!\w)', r'\1', text)

    # ── 19. Listes ** *** → tirets ───────────────────────────────
    text = re.sub(r'^\*{3}\s+', '      - ', text, flags=re.MULTILINE)
    text = re.sub(r'^\*{2}\s+', '   - ',    text, flags=re.MULTILINE)
    text = re.sub(r'^\*{1}\s+', '- ',       text, flags=re.MULTILINE)

    # ── 20. Listes ## ### → tirets ───────────────────────────────
    text = re.sub(r'^#{3}\s+', '      - ', text, flags=re.MULTILINE)
    text = re.sub(r'^#{2}\s+', '   - ',    text, flags=re.MULTILINE)
    text = re.sub(r'^#{1}\s+', '- ',       text, flags=re.MULTILINE)

    # ── 21. Tableaux ─────────────────────────────────────────────
    def clean_table_row(match):
        row   = match.group(0)
        row   = re.sub(r'\|[_/\\<>=^~\d.]*\.', '|', row)
        cells = [c.strip() for c in row.split('|') if c.strip()]
        return ' | '.join(cells)

    text = re.sub(r'^\|.+\|$', clean_table_row, text, flags=re.MULTILINE)
    text = re.sub(r'^table\{[^}]*\}\.$', '', text, flags=re.MULTILINE)

    # ── 22. Footnotes ────────────────────────────────────────────
    text = re.sub(r'^fn\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\d+\]',     '', text)

    # ── 23. Règle horizontale ────────────────────────────────────
    text = re.sub(r'^-{4,}$', '', text, flags=re.MULTILINE)

    # ── 24. Liens Redmine spéciaux ───────────────────────────────
    text = re.sub(r'##(\d+)',                r'ticket #\1',        text)
    text = re.sub(r'source:"?([^"\s]+)"?',   r'fichier \1',        text)
    text = re.sub(r'document:"?([^"\s]+)"?', r'document \1',       text)
    text = re.sub(r'version:"?([^"\s]+)"?',  r'version \1',        text)
    text = re.sub(r'commit:[^\s]+',          '',                   text)

    # ── 25. Balises HTML résiduelles ─────────────────────────────
    text = re.sub(r'<[^>]+>', '', text)

    # ── 26. Entités HTML ─────────────────────────────────────────
    html_entities = {
        '&nbsp;': ' ', '&amp;':  '&', '&lt;':   '<',
        '&gt;':   '>', '&quot;': '"', '&#42;':  '*',
        '&#95;':  '_', '&#64;':  '@', '&mdash;':'—',
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # ── 27. Nettoyage final ───────────────────────────────────────
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}',  ' ',    text)
    text = re.sub(r'[ \t]+$', '',    text, flags=re.MULTILINE)

    return text.strip()

def _merge_small_chunks(
    chunks:    list[dict],
    min_chars: int = None,
    min_words: int = None,
) -> list[dict]:
    """
    Fusionne les chunks trop petits avec leur voisin suivant.
    Robuste : fonctionne avec ou sans clés section/h1/h2/h3.
    """
    min_chars = min_chars or settings.min_chunk_chars
    min_words = min_words or settings.min_chunk_words

    if len(chunks) <= 1:
        return chunks

    def _is_small(chunk: dict) -> bool:
        text  = chunk.get("text", "").strip()
        words = [w for w in text.split() if len(w) > 1]
        return len(text) < min_chars or len(words) < min_words

    result = []
    i      = 0

    while i < len(chunks):
        current = chunks[i]

        if _is_small(current) and i + 1 < len(chunks):
            nxt = chunks[i + 1]

            # ── Robuste : .get() avec défaut "" ───────────────────
            section = nxt.get("section", "") or current.get("section", "")
            h1      = nxt.get("h1", "")     or current.get("h1", "")
            h2      = nxt.get("h2", "")     or current.get("h2", "")
            h3      = nxt.get("h3", "")     or current.get("h3", "")

            fused = {
                "text":    current["text"].strip() + "\n\n" + nxt["text"].strip(),
                "section": section,
                "h1":      h1,
                "h2":      h2,
                "h3":      h3,
            }

            logger.debug(
                f"_merge_small_chunks | fusion | "
                f"'{current['text'][:40]}' + '{nxt['text'][:40]}'"
            )

            chunks = chunks[:i] + [fused] + chunks[i + 2:]
            continue

        result.append(current)
        i += 1

    # Dernier chunk encore petit → fusionner avec précédent
    if len(result) >= 2 and _is_small(result[-1]):
        last = result.pop()
        prev = result[-1]
        result[-1] = {
            "text":    prev["text"].strip() + "\n\n" + last["text"].strip(),
            "section": prev.get("section", "") or last.get("section", ""),
            "h1":      prev.get("h1", "")     or last.get("h1", ""),
            "h2":      prev.get("h2", "")     or last.get("h2", ""),
            "h3":      prev.get("h3", "")     or last.get("h3", ""),
        }

    return result

def clean_text(text: str) -> str:
    """
    Nettoyage basique sans Textile.
    Utilisé uniquement si le texte est déjà propre (pas de Textile).
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}",  " ",    text)

    return text.strip()


def textile_to_markdown(text: str) -> str:
    if not text:
        return ""

    # ── Headers (AVANT les listes pour éviter collision) ──────────
    text = re.sub(r"^h1\.\s*(.+)$", r"# \1",      text, flags=re.MULTILINE)
    text = re.sub(r"^h2\.\s*(.+)$", r"## \1",     text, flags=re.MULTILINE)
    text = re.sub(r"^h3\.\s*(.+)$", r"### \1",    text, flags=re.MULTILINE)
    text = re.sub(r"^h4\.\s*(.+)$", r"#### \1",   text, flags=re.MULTILINE)
    text = re.sub(r"^h5\.\s*(.+)$", r"##### \1",  text, flags=re.MULTILINE)
    text = re.sub(r"^h6\.\s*(.+)$", r"###### \1", text, flags=re.MULTILINE)

    # ── Listes numérotées Textile ## ### → indentées ───────────────
    # CRITIQUE : convertir AVANT les listes simples #
    text = re.sub(r"^#{3}\s+(.+)$", r"      1. \1", text, flags=re.MULTILINE)
    text = re.sub(r"^#{2}\s+(.+)$", r"   1. \1",    text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.+)$",    r"1. \1",       text, flags=re.MULTILINE)

    # ── Listes non numérotées Textile ** *** ──────────────────────
    text = re.sub(r"^\*{3}\s+(.+)$", r"      - \1", text, flags=re.MULTILINE)
    text = re.sub(r"^\*{2}\s+(.+)$", r"   - \1",    text, flags=re.MULTILINE)
    text = re.sub(r"^\*\s+(.+)$",    r"- \1",       text, flags=re.MULTILINE)

    # ── Gras *bold* → **bold** ────────────────────────────────────
    # Seulement si pas déjà converti en liste
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"**\1**", text)

    # ── Liens "texte":url → texte ─────────────────────────────────
    text = re.sub(r'"([^"]+)":\S+', r"\1", text)

    # ── Monospace @code@ → `code` ─────────────────────────────────
    text = re.sub(r"@(.+?)@", r"`\1`", text)

    # ── Blocs pre → code fences ───────────────────────────────────
    text = re.sub(
        r'<pre>(.*?)</pre>',
        lambda m: f"\n```\n{m.group(1).strip()}\n```\n",
        text,
        flags=re.DOTALL,
    )

    return text

# ─────────────────────────────────────────────
#   SPLITTERS LANGCHAIN
# ─────────────────────────────────────────────

def get_recursive_splitter(
    chunk_size:    int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> RecursiveCharacterTextSplitter:
    """
    Splitter récursif avec séparateurs adaptés au contenu Redmine.

    Ordre des séparateurs (du plus fort au plus faible) :
    1. Triple saut de ligne  → séparation très forte
    2. Double saut de ligne  → séparation de paragraphe
    3. Saut de ligne simple  → séparation de ligne
    4. Fin de phrase         → séparation sémantique
    5. Virgule / point-virgule
    6. Espace                → dernier recours
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=    chunk_size,
        chunk_overlap= chunk_overlap,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
    )


def get_markdown_splitter() -> MarkdownHeaderTextSplitter:
    """
    Splitter Markdown qui découpe selon les headers.
    Retourne des Documents avec metadata {"h1": ..., "h2": ..., "h3": ...}
    """
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )


# ─────────────────────────────────────────────
#   TÉLÉCHARGEMENT FICHIERS
# ─────────────────────────────────────────────

async def download_file(url: str, api_key: str) -> bytes | None:
    """
    Télécharge un fichier depuis Redmine.
    """
    ssl_verify = get_ssl_verify()          

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            verify= ssl_verify,
        ) as client:
            response = await client.get(
                url,
                headers={"X-Redmine-API-Key": api_key},
            )

            if response.status_code != 200:
                logger.warning(
                    f"download_file | {response.status_code} | {url}"
                )
                return None

            content = response.content

            if len(content) > settings.max_file_size:
                logger.warning(
                    f"download_file | fichier trop grand | "
                    f"{len(content)} bytes | {url}"
                )
                return None

            logger.debug(
                f"download_file | OK | {len(content)} bytes | {url}"
            )
            return content

    except Exception as e:
        logger.error(f"download_file | erreur | {url} | {str(e)}")
        return None


# ─────────────────────────────────────────────
#   EXTRACTION CONTENU FICHIERS
# ─────────────────────────────────────────────

def extract_pdf(content: bytes) -> str:
    """
    Extrait le texte d'un PDF via PyMuPDF.
    """
    try:
        doc   = fitz.open(stream=content, filetype="pdf")
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text.strip()}")

        doc.close()
        result = "\n\n".join(pages)
        logger.debug(
            f"extract_pdf | {len(pages)} pages | {len(result)} chars"
        )
        return result

    except Exception as e:
        logger.error(f"extract_pdf | erreur | {str(e)}")
        return ""


def extract_docx(content: bytes) -> str:
    """
    Extrait le texte d'un fichier DOCX.
    """
    try:
        document   = docx.Document(io.BytesIO(content))
        paragraphs = []

        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                )
                if row_text:
                    paragraphs.append(row_text)

        result = "\n\n".join(paragraphs)
        logger.debug(
            f"extract_docx | {len(paragraphs)} paragraphes | {len(result)} chars"
        )
        return result

    except Exception as e:
        logger.error(f"extract_docx | erreur | {str(e)}")
        return ""


def extract_xlsx(content: bytes) -> str:
    """
    Extrait le texte d'un fichier XLSX.
    """
    try:
        wb    = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        lines = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            lines.append(f"Feuille : {sheet_name}")

            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join(
                    str(cell)
                    for cell in row
                    if cell is not None and str(cell).strip()
                )
                if row_text:
                    lines.append(row_text)

        result = "\n".join(lines)
        logger.debug(
            f"extract_xlsx | {len(wb.sheetnames)} feuilles | {len(result)} chars"
        )
        return result

    except Exception as e:
        logger.error(f"extract_xlsx | erreur | {str(e)}")
        return ""


def extract_text_plain(content: bytes) -> str:
    """
    Extrait le texte d'un fichier texte brut.
    """
    try:
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return ""

    except Exception as e:
        logger.error(f"extract_text_plain | erreur | {str(e)}")
        return ""


async def extract_attachment_content(
    attachment: dict,
    api_key:    str,
) -> str:
    """
    Télécharge et extrait le contenu textuel d'une pièce jointe.
    """
    filename     = attachment.get("filename",     "")
    content_type = attachment.get("content_type", "")
    content_url  = attachment.get("content_url",  "")
    filesize     = attachment.get("filesize",      0)

    if content_type not in SUPPORTED_TEXT_TYPES:
        logger.debug(
            f"extract_attachment_content | type non supporté | "
            f"{content_type} | {filename}"
        )
        return ""
    
    if filename.lower().endswith(".csv") or content_type == "text/csv":
        logger.info(
            f"extract_attachment_content | CSV ignoré | {filename}"
        )
        return ""
    if filename.lower().endswith(".xlsx") :
        logger.info(
            f"extract_attachment_content | xlsx ignoré | {filename}"
        )
        return ""
    if filename.lower().endswith(".log") :
        logger.info(
            f"extract_attachment_content | logs ignoré | {filename}"
        )
        return ""
    if filesize > settings.max_file_size:
        logger.warning(
            f"extract_attachment_content | fichier trop grand | "
            f"{filesize} bytes | {filename}"
        )
        return ""

    if not content_url:
        return ""

    logger.info(
        f"extract_attachment_content | téléchargement | "
        f"{filename} | {content_type}"
    )

    content = await download_file(content_url, api_key)

    if not content:
        return ""

    if content_type == "application/pdf":
        return extract_pdf(content)

    
    #elif content_type == (
    #    "application/vnd.openxmlformats-officedocument"
    #    ".wordprocessingml.document"
    #):
    #    return extract_docx(content)

    #elif content_type == (
    #    "application/vnd.openxmlformats-officedocument"
    #    ".spreadsheetml.sheet"
    #):
    #    return extract_xlsx(content)

    #elif content_type in {
    #    "text/plain", "text/html", "text/markdown"
    #}:
    #    return extract_text_plain(content)

    return ""


# ─────────────────────────────────────────────
#   HELPERS INTERNES
# ─────────────────────────────────────────────

def _build_section_path(metadata: dict) -> str:
    """
    Construit le chemin de section depuis les metadata LangChain.

    Exemple : {"h1": "Installation", "h2": "Configuration"}
           → "Installation > Configuration"
    """
    parts = []
    for key in ["h1", "h2", "h3"]:
        val = metadata.get(key, "").strip()
        if val:
            # Nettoyer le préfixe markdown si strip_headers=False
            val = re.sub(r"^#+\s*", "", val).strip()
            if val:
                parts.append(val)
    return " > ".join(parts)


def _split_long_text(
    text:          str,
    chunk_size:    int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Re-découpe un texte trop long avec le splitter récursif.
    Retourne toujours une liste non vide si text non vide.
    """
    if not text.strip():
        return []

    splitter = get_recursive_splitter(chunk_size, chunk_overlap)
    chunks   = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


def _build_issue_header(issue: dict) -> str:
    """
    En-tête compact d'un ticket.

    Répété dans CHAQUE chunk du ticket pour que le LLM
    sache toujours de quel ticket il parle, même sans contexte.

    Format court pour ne pas gaspiller de tokens.
    """
    issue_id    = issue.get("id",          "")
    subject     = issue.get("subject",     "") or ""
    status      = issue.get("status",      {}) or {}
    priority    = issue.get("priority",    {}) or {}
    tracker     = issue.get("tracker",     {}) or {}
    assigned_to = issue.get("assigned_to", {}) or {}
    project     = issue.get("project",     {}) or {}

    return (
        f"Ticket #{issue_id} : {subject}\n"
        f"Projet: {project.get('name', '')} | "
        f"Tracker: {tracker.get('name', '')} | "
        f"Statut: {status.get('name', '')} | "
        f"Priorité: {priority.get('name', '')} | "
        f"Assigné: {assigned_to.get('name', 'Non assigné')}"
    )


# ─────────────────────────────────────────────
#   WIKI CHUNKS
# ─────────────────────────────────────────────
async def build_wiki_chunks(
    wiki_page:    dict,
    project_name: str,
    project_id:   str,
    api_key:      str = None,
) -> list[dict]:
    title       = wiki_page.get("title",      "")
    raw_text    = wiki_page.get("text",        "") or ""
    version     = wiki_page.get("version",     1)
    updated_on  = wiki_page.get("updated_on",  "")
    created_on  = wiki_page.get("created_on",  "")
    author      = wiki_page.get("author",      {}) or {}
    author_name = author.get("name", "")
    author_id   = author.get("id",   "")
    attachments = wiki_page.get("attachments", []) or []

    if not raw_text.strip() and not attachments:
        logger.warning(f"build_wiki_chunks | page vide | title={title}")
        return []

    result = []

    if raw_text.strip():

        md_text = textile_to_markdown(raw_text)

        if title:
            md_text = f"# {title}\n\n{md_text}"

        md_splitter = get_markdown_splitter()

        try:
            header_docs = md_splitter.split_text(md_text)
        except Exception as e:
            logger.error(
                f"build_wiki_chunks | markdown split erreur | "
                f"title={title} | {str(e)}"
            )
            header_docs = []

        raw_chunks = []

        if header_docs:
            for doc in header_docs:
                content = doc.page_content.strip()
                if not content:
                    continue

                meta    = doc.metadata
                section = _build_section_path(meta)

                if len(content) > settings.chunk_size:
                    sub_texts = _split_long_text(content)
                    for sub in sub_texts:
                        if sub.strip():
                            raw_chunks.append({
                                "text":    sub,
                                "section": section,
                                "h1":      meta.get("h1", ""),
                                "h2":      meta.get("h2", ""),
                                "h3":      meta.get("h3", ""),
                            })
                else:
                    # ── Pas de filtre ici, _merge s'en charge ──────
                    raw_chunks.append({
                        "text":    content,
                        "section": section,
                        "h1":      meta.get("h1", ""),
                        "h2":      meta.get("h2", ""),
                        "h3":      meta.get("h3", ""),
                    })

        else:
            # ── Fallback : pas de headers détectés ────────────────
            fallback_texts = _split_long_text(md_text)
            for t in fallback_texts:
                if t.strip():           # ← seulement les vrais vides
                    raw_chunks.append({
                        "text":    t,
                        "section": "",
                        "h1":      "",
                        "h2":      "",
                        "h3":      "",
                    })

        # ── Fusion → dédoublonnage → nettoyage final ──────────────
        raw_chunks = _merge_small_chunks(raw_chunks)
        raw_chunks = _deduplicate_chunks(raw_chunks)
        raw_chunks = [c for c in raw_chunks if c.get("text", "").strip()]

        total = len(raw_chunks)

        for i, chunk in enumerate(raw_chunks):
            text       = chunk["text"]
            word_count = len(text.split())

            result.append({
                "text": text,
                "metadata": {
                    "source":          "wiki",
                    "project_name":    project_name,
                    "project_id":      str(project_id),
                    "title":           title,
                    "url":             f"/projects/{project_id}/wiki/{title}",
                    "content_type":    "wiki_text",
                    "section":         chunk["section"],
                    "h1":              re.sub(r"^#+\s*", "", chunk["h1"]).strip(),
                    "h2":              re.sub(r"^#+\s*", "", chunk["h2"]).strip(),
                    "h3":              re.sub(r"^#+\s*", "", chunk["h3"]).strip(),
                    "chunk_index":     i,
                    "total_chunks":    total,
                    "is_first":        i == 0,
                    "is_last":         i == total - 1,
                    "author_name":     author_name,
                    "author_id":       author_id,
                    "version":         version,
                    "created_on":      created_on,
                    "updated_on":      updated_on,
                    "word_count":      word_count,
                    "char_count":      len(text),
                    "has_code":        "`" in text or "```" in text,
                    "has_table":       "|" in text,
                    "has_list":        bool(
                        re.search(r"^[-*1]\s", text, re.MULTILINE)
                    ),
                    "has_attachments": len(attachments) > 0,
                },
            })

    # ── Pièces jointes (inchangé) ─────────────────────────────────
    for attachment in attachments:
        filename     = attachment.get("filename",     "")
        content_type = attachment.get("content_type", "")
        content_url  = attachment.get("content_url",  "")
        filesize     = attachment.get("filesize",      0)
        attach_id    = attachment.get("id",            "")

        attach_metadata = {
            "source":        "wiki_attachment",
            "project_name":  project_name,
            "project_id":    str(project_id),
            "wiki_title":    title,
            "attachment_id": str(attach_id),
            "filename":      filename,
            "content_type":  content_type,
            "filesize":      filesize,
            "content_url":   content_url,
            "updated_on":    updated_on,
            "has_content":   False,
            "chunk_index":   0,
            "total_chunks":  1,
            "url":           f"/projects/{project_id}/wiki/{title}",
        }

        if api_key and content_type in SUPPORTED_TEXT_TYPES:
            file_content = await extract_attachment_content(attachment, api_key)

            if file_content.strip():
                file_text   = (
                    f"Contenu du fichier '{filename}' "
                    f"(wiki: {title}) :\n\n{file_content}"
                )
                file_chunks = _split_long_text(file_text)

                for i, chunk in enumerate(file_chunks):
                    if chunk.strip():
                        result.append({
                            "text": chunk,
                            "metadata": {
                                **attach_metadata,
                                "has_content":  True,
                                "chunk_index":  i,
                                "total_chunks": len(file_chunks),
                                "word_count":   len(chunk.split()),
                                "char_count":   len(chunk),
                            },
                        })

                logger.info(
                    f"build_wiki_chunks | attachment indexé | "
                    f"{filename} | {len(file_chunks)} chunks"
                )
                continue

        meta_text = (
            f"Pièce jointe de la page wiki '{title}' :\n"
            f"Fichier : {filename}\n"
            f"Type : {content_type}\n"
            f"Taille : {filesize} bytes"
        )
        result.append({"text": meta_text, "metadata": attach_metadata})

    logger.info(
        f"build_wiki_chunks | title={title} | "
        f"project={project_id} | {len(result)} chunks total"
    )
    return result


# ─────────────────────────────────────────────
#   ISSUE CHUNKS
# ─────────────────────────────────────────────
def extract_custom_fields(issue: dict) -> tuple[dict, str]:
    """
    Extrait les champs personnalisés Redmine.
    
    Retourne :
    - metadata_dict : dict des custom fields pour les métadonnées Qdrant
    - text_block    : texte formaté à inclure dans le chunk
    
    Gère les types :
    - string  : valeur simple
    - list    : valeurs multiples (ex: sprints)
    - None/'' : ignoré
    """
    custom_fields = issue.get("custom_fields", []) or []

    metadata_dict = {}
    text_lines    = []

    for field in custom_fields:
        name  = field.get("name",  "") or ""
        value = field.get("value", "")
        fid   = field.get("id",    "")

        # ── Ignorer les champs vides ───────────────────────────────────
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue

        # ── Normaliser la clé pour les métadonnées ────────────────────
        meta_key = (
            name.lower()
                .strip()
                .replace(" ", "_")
                .replace("'", "")
                .replace("é", "e")
                .replace("è", "e")
                .replace("ê", "e")
                .replace("à", "a")
                .replace("ù", "u")
                .replace("ô", "o")
                .replace("î", "i")
                .replace("ç", "c")
                .replace("/", "_")
                .replace("-", "_")
        )
        meta_key = f"cf_{meta_key}"

        # ── Normaliser la valeur ──────────────────────────────────────
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value if v)
        else:
            value_str = str(value).strip()

        if not value_str:
            continue

        # ── Stocker dans metadata ──────────────────────────────────────
        metadata_dict[meta_key] = value_str

        # ── Ajouter au texte ───────────────────────────────────────────
        text_lines.append(f"{name} : {value_str}")

    text_block = "\n".join(text_lines)

    return metadata_dict, text_block

async def build_issue_chunks(
    issue:   dict,
    api_key: str = None,
) -> list[dict]:
    """
    Prépare les chunks d'un ticket pour l'indexation.

    - Description      → chunk(s) séparé(s)
    - Chaque commentaire → chunk séparé
    - En-tête répété dans chaque chunk
    - Métadonnées enrichies : component, comment_by, comment_date
    """
    issue_id    = issue.get("id",          "")
    subject     = issue.get("subject",     "") or ""
    description = issue.get("description", "") or ""
    done_ratio  = issue.get("done_ratio",  0)
    start_date  = issue.get("start_date",  "")
    due_date    = issue.get("due_date",    "")
    created_on  = issue.get("created_on",  "")
    updated_on  = issue.get("updated_on",  "")
    estimated_h = issue.get("estimated_hours", "")
    spent_h     = issue.get("spent_hours",     "")
    attachments = issue.get("attachments", []) or []
    journals    = issue.get("journals",    []) or []
    children    = issue.get("children",    []) or []
    relations   = issue.get("relations",   []) or []

    custom_metadata, custom_text = extract_custom_fields(issue)
    project     = issue.get("project",     {}) or {}
    status      = issue.get("status",      {}) or {}
    priority    = issue.get("priority",    {}) or {}
    tracker     = issue.get("tracker",     {}) or {}
    assigned_to = issue.get("assigned_to", {}) or {}
    author      = issue.get("author",      {}) or {}

    project_id       = str(project.get("id", ""))
    project_name     = project.get("name",        "")
    status_name      = status.get("name",         "")
    priority_name    = priority.get("name",       "")
    tracker_name     = tracker.get("name",        "")
    assigned_to_name = assigned_to.get("name",    "Non assigné")
    assigned_to_id   = assigned_to.get('id',      'Non assigne')
    author_name      = author.get("name",         "")
    author_id        = author.get('id',           '')

    # ── En-tête compact répété dans chaque chunk ──────────────────────────────
    header = _build_issue_header(issue)

    # ── Métadonnées de base communes à tous les chunks ────────────────────────
    base_metadata = {
        "source":          "issue",
        "project_id":      str(project_id),
        "project_name":    project_name,
        "issue_id":        str(issue_id),
        "subject":         subject,
        "status":          status_name,
        "tracker":         tracker_name,
        "priority":        priority_name,
        "assigned_to_name": assigned_to_name,
        'assigned_to_id'  : assigned_to_id,    
        "author_name":     author_name,
        'author_id':       author_id,   
        "done_ratio":      done_ratio,
        "start_date":      start_date,
        "due_date":        due_date,
        "created_on":      created_on,
        "updated_on":      updated_on,
        "estimated_hours": str(estimated_h) if estimated_h else "",
        "spent_hours":     str(spent_h)     if spent_h     else "",
        "has_attachments": len(attachments) > 0,
        "url":             f"/issues/{issue_id}",
        "content_type":    "issue_text",
        **custom_metadata
    }

    result = []

    # ── CHUNK(S) DESCRIPTION ──────────────────────────────────────────────────
    if description.strip():

        desc_lines = []
        desc_lines.append(header)
        desc_lines.append("")

        if start_date:
            desc_lines.append(f"Date début : {start_date}")
        if due_date:
            desc_lines.append(f"Date fin : {due_date}")
        if estimated_h:
            desc_lines.append(f"Heures estimées : {estimated_h}h")
        if spent_h:
            desc_lines.append(f"Heures passées : {spent_h}h")
        if done_ratio:
            desc_lines.append(f"Avancement : {done_ratio}%")

        if custom_text:
            desc_lines.append(f"\nInformations complémentaires :\n{custom_text}")

        if children:
            desc_lines.append(
                "Sous-tickets : " +
                ", ".join(
                    f"#{c.get('id')} {c.get('subject', '')}"
                    for c in children
                )
            )

        if relations:
            for rel in relations:
                desc_lines.append(
                    f"Relation {rel.get('relation_type', '')} "
                    f"→ #{rel.get('issue_to_id', '')}"
                )
                
        desc_lines.append(f"\nDescription :\n{clean_textile(description)}")

        desc_full = "\n".join(desc_lines)

        if len(desc_full) > settings.chunk_size:
            sub_texts = _split_long_text(desc_full)
        else:
            sub_texts = [desc_full]

        for i, sub in enumerate(sub_texts):
            if _is_valid_chunk(sub):
                result.append({
                    "text": sub,
                    "metadata": {
                        **base_metadata,
                        "component":    "description",
                        "chunk_index":  i,
                        "total_chunks": len(sub_texts),
                        "word_count":   len(sub.split()),
                        "char_count":   len(sub),
                    },
                })

    # ── 1 CHUNK PAR COMMENTAIRE ───────────────────────────────────────────────
    for journal in journals:
        notes = journal.get("notes", "") or ""

        if not notes.strip():
            continue

        notes_clean = clean_textile(notes)

        if not notes_clean.strip():          
            logger.debug(
                f"build_issue_chunks | commentaire vide après nettoyage | "
                f"#{issue_id}"
            )
            continue
        # ── PAS de filtre sur notes brutes ────────────────────────
        j_author = journal.get("user",       {}) or {}
        j_name   = j_author.get("name",      "")
        j_date   = journal.get("created_on", "")

        comment_text = (
            f"{header}\n"
            f"Commentaire de {j_name} ({j_date}) :\n"
            f"{notes_clean}"
        )

        # ── Tester sur le texte COMPLET (avec header) ─────────────
        if not _is_valid_chunk(comment_text):
            logger.debug(
                f"build_issue_chunks | commentaire ignoré après construction | "
                f"#{issue_id} | '{notes[:50]}'"
            )
            continue

        if len(comment_text) > settings.chunk_size:
            sub_texts = _split_long_text(comment_text)
        else:
            sub_texts = [comment_text]

        for i, sub in enumerate(sub_texts):
            if sub.strip():
                result.append({
                    "text": sub,
                    "metadata": {
                        **base_metadata,
                        "component":    "comment",
                        "comment_by":   j_name,
                        "comment_date": j_date,
                        "chunk_index":  i,
                        "total_chunks": len(sub_texts),
                        "word_count":   len(sub.split()),
                        "char_count":   len(sub),
                    },
                })

    # ── Fallback : header seul si aucun contenu ───────────────────────────────
    if not result:
        result.append({
            "text": header,
            "metadata": {
                **base_metadata,
                "component":    "header_only",
                "chunk_index":  0,
                "total_chunks": 1,
                "word_count":   len(header.split()),
                "char_count":   len(header),
            },
        })

    # ── PIÈCES JOINTES ────────────────────────────────────────────────────────
    for attachment in attachments:
        filename     = attachment.get("filename",     "")
        content_type = attachment.get("content_type", "")
        content_url  = attachment.get("content_url",  "")
        filesize     = attachment.get("filesize",      0)
        attach_id    = attachment.get("id",            "")

        attach_metadata = {
            "source":        "issue_attachment",
            "project_id":    str(project_id),
            "project_name":  project_name,
            "issue_id":      str(issue_id),
            "subject":       subject,
            "attachment_id": str(attach_id),
            "filename":      filename,
            "content_type":  content_type,
            "filesize":      filesize,
            "content_url":   content_url,
            "updated_on":    updated_on,
            "has_content":   False,
            "chunk_index":   0,
            "total_chunks":  1,
            "url":           f"/issues/{issue_id}",
        }

        # ── Tenter d'extraire le contenu ──────────────────────────────────────
        if api_key and content_type in SUPPORTED_TEXT_TYPES:
            file_content = await extract_attachment_content(
                attachment, api_key
            )

            if file_content.strip():
                file_text = (
                    f"Contenu du fichier '{filename}' "
                    f"(ticket #{issue_id} : {subject}) :\n\n"
                    f"{file_content}"
                )

                file_chunks = _split_long_text(file_text)

                for i, chunk in enumerate(file_chunks):
                    if _is_valid_chunk(chunk):  
                        result.append({
                            "text": chunk,
                            "metadata": {
                                **attach_metadata,
                                "has_content":  True,
                                "chunk_index":  i,
                                "total_chunks": len(file_chunks),
                                "word_count":   len(chunk.split()),
                                "char_count":   len(chunk),
                            },
                        })

                logger.info(
                    f"build_issue_chunks | attachment indexé | "
                    f"{filename} | {len(file_chunks)} chunks"
                )
                continue

        # ── Fallback : métadonnées seulement ──────────────────────────────────
        meta_text = (
            f"Pièce jointe du ticket #{issue_id} '{subject}' :\n"
            f"Fichier : {filename}\n"
            f"Type : {content_type}\n"
            f"Taille : {filesize} bytes"
        )

        result.append({
            "text":     meta_text,
            "metadata": attach_metadata,
        })

    logger.info(
        f"build_issue_chunks | #{issue_id} | "
        f"project={project_id} | {len(result)} chunks total"
    )
    return result

# ─────────────────────────────────────────────
#   PROJECT CHUNKS
# ─────────────────────────────────────────────

def build_project_chunks(project: dict) -> list[dict]:
    """
    Prépare les chunks d'un projet pour l'indexation.
    """
    project_id  = str(project.get("id", ""))
    name        = project.get("name",        "") or ""
    description = project.get("description", "") or ""
    status      = project.get("status",      1)
    is_public   = project.get("is_public",   False)
    created_on  = project.get("created_on",  "")
    updated_on  = project.get("updated_on",  "")

    trackers   = project.get("trackers",         []) or []
    categories = project.get("issue_categories", []) or []
    modules    = project.get("enabled_modules",  []) or []

    trackers_text   = ", ".join(t.get("name", "") for t in trackers)
    categories_text = ", ".join(c.get("name", "") for c in categories)
    modules_text    = ", ".join(m.get("name", "") for m in modules)

    lines = []
    lines.append(f"Projet : {name} ({project_id})")
    lines.append(
        f"Statut : {'Actif' if status == 1 else 'Archivé'}"
    )
    lines.append(
        f"Visibilité : {'Public' if is_public else 'Privé'}"
    )

    if description.strip():
        lines.append(f"\nDescription :\n{clean_textile(description)}")
    if trackers_text:
        lines.append(f"\nTrackers disponibles : {trackers_text}")
    if categories_text:
        lines.append(f"Catégories : {categories_text}")
    if modules_text:
        lines.append(f"Modules activés : {modules_text}")

    full_text = "\n".join(lines)

    if not full_text.strip():
        logger.warning(
            f"build_project_chunks | projet vide | id={project_id}"
        )
        return []

    chunks = _split_long_text(full_text)
    result = []

    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "metadata": {
                "source":       "project",
                "project_id":   str(project_id),
                "name":         name,
                "status":       "actif" if status == 1 else "archivé",
                "is_public":    is_public,
                "created_on":   created_on,
                "updated_on":   updated_on,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "url":          f"/projects/{project_id}",
                "content_type": "project_text",
                "word_count":   len(chunk.split()),
                "char_count":   len(chunk),
            },
        })

    logger.info(
        f"build_project_chunks | {name} ({project_id}) | "
        f"{len(result)} chunks"
    )
    return result