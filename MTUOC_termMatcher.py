import sqlite3
import json
import csv
import sys
import xml.etree.ElementTree as ET
import spacy
from spacy.util import is_package
from rapidfuzz import process, fuzz
from langcodes import Language, LanguageTagError

class SpacyMTUOCTokenizer:
    """
    Custom spaCy tokenizer integration for MTUOC.
    Handles smart loading of full pipeline models or blank language fallbacks.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.joiner = "￭"
        self.splitter = "▁"
        self.nlp = self._load_or_download_model(model_name)

    def _load_or_download_model(self, model_name):
        # 1. Direct load attempt
        if is_package(model_name):
            try:
                return spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
            except Exception:
                pass

        # 2. If it's a full model name, try to download it
        if "_" in model_name:
            print(f"Model '{model_name}' not found. Downloading...", file=sys.stderr)
            try:
                spacy.cli.download(model_name)
                return spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
            except Exception as e:
                print(f"Download failed for '{model_name}': {e}", file=sys.stderr)

        # 3. Fallback to blank language model (e.g., 'ca', 'en', 'is')
        try:
            return spacy.blank(model_name)
        except Exception:
            print(f"Error: '{model_name}' is not a valid model or language code. Defaulting to 'en'.", file=sys.stderr)
            return spacy.blank("en")

    def tokenize(self, text, mode="tokenize"):
        doc = self.nlp(text)
        if mode == "tokenize":
            return " ".join([t.text for t in doc])
        return text


class termMatcher:
    def __init__(self, db_path="terminology.db", max_ngram=4):
        """
        Initializes the termMatcher class, establishes a connection to the SQLite database,
        and sets the default maximum n-gram size for the sliding window search.
        """
        self.db_path = db_path
        self.max_ngram = max_ngram
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        self.tokenizers = {}  # Cache to reuse spaCy tokenizers for different languages

    def _create_tables(self):
        """Creates the necessary tables for concepts and terms."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                definitions TEXT,
                global_metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT,
                language TEXT NOT NULL,
                term TEXT NOT NULL,
                part_of_speech TEXT,
                term_metadata TEXT,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        """)
        self.conn.commit()

    def _normalize_language_code(self, language_code, for_spacy=False):
        """
        Smart language code normalizer using the 'langcodes' library.
        Handles conversions like 'eng' -> 'en', 'en-US' -> 'en', 'Catalan' -> 'ca'.
        """
        if not language_code:
            return "en"
            
        # If it's a specific spaCy model pipeline name (like ca_core_news_sm), don't touch it
        if "_" in str(language_code) and "news" in str(language_code):
            return language_code
            
        try:
            # First try Language.get() which handles direct codes like 'en', 'en-US', 'ca-ES' flawlessly
            lang_obj = Language.get(language_code)
        except (LanguageTagError, ValueError):
            try:
                # If .get() fails, fallback to .find() for natural names like 'English' or 'Catalan'
                lang_obj = Language.find(language_code)
            except (LookupError, LanguageTagError, ValueError):
                # Fallback if everything fails
                clean_code = str(language_code).strip()
                if "-" in clean_code:
                    return clean_code.split("-")[0] if for_spacy else clean_code
                return clean_code[:2] if for_spacy else clean_code

        # Extract the correct string based on the destination target
        if for_spacy:
            # spaCy blank models only accept the primary 2-letter subtag (e.g., 'ca', 'en')
            return lang_obj.language
        else:
            # For DB lookups, standardizing as a clean standard string (e.g., 'en-US' -> 'en_US' or 'en')
            return str(lang_obj)

    def _get_tokenizer(self, language_code):
        """Gets or creates a SpacyMTUOCTokenizer instance using the 2-letter code required by spaCy."""
        spacy_code = self._normalize_language_code(language_code, for_spacy=True)
        if spacy_code not in self.tokenizers:
            self.tokenizers[spacy_code] = SpacyMTUOCTokenizer(spacy_code)
        return self.tokenizers[spacy_code]

    def load_tbx(self, file_path):
        """Parses a TBX file and extracts concepts, terms, definitions, and metadata."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        cursor = self.conn.cursor()

        for term_entry in root.findall('.//{*}termEntry'):
            concept_id = term_entry.get('id') or term_entry.get('{http://www.w3.org/XML/1998/namespace}id')
            if not concept_id:
                continue
            
            definitions = {}
            global_metadata = {}
            
            for descrip in term_entry.findall('{*}descrip'):
                descrip_type = descrip.get('type')
                if descrip_type == 'subjectField':
                    global_metadata['subject_field'] = descrip.text
                elif descrip_type == 'definition':
                    definitions['global'] = descrip.text

            for child in term_entry:
                tag_local_name = child.tag.split('}')[-1]
                if tag_local_name not in ['langSet', 'descrip']:
                    global_metadata[tag_local_name] = child.text or ET.tostring(child, encoding='utf-8').decode('utf-8')

            cursor.execute(
                "INSERT OR REPLACE INTO concepts (id, definitions, global_metadata) VALUES (?, ?, ?)",
                (concept_id, json.dumps(definitions), json.dumps(global_metadata))
            )

            for lang_set in term_entry.findall('.//{*}langSet'):
                lang = lang_set.get('{http://www.w3.org/XML/1998/namespace}lang') or lang_set.get('lang')
                if not lang:
                    continue
                
                # Standardize language tag for the database
                lang = self._normalize_language_code(lang, for_spacy=False)
                
                for descrip in lang_set.findall('{*}descrip'):
                    if descrip.get('type') == 'definition':
                        definitions[lang] = descrip.text
                
                cursor.execute("UPDATE concepts SET definitions = ? WHERE id = ?", (json.dumps(definitions), concept_id))

                for tig in lang_set.findall('.//{*}tig') or lang_set.findall('.//{*}ntig') or [lang_set]:
                    term_element = tig.find('.//{*}term')
                    if term_element is None or not term_element.text:
                        continue
                        
                    term_text = term_element.text.strip()
                    term_metadata = {}
                    part_of_speech = None
                    
                    for term_note in tig.findall('.//{*}termNote'):
                        note_type = term_note.get('type')
                        if note_type == 'partOfSpeech':
                            part_of_speech = term_note.text
                        else:
                            term_metadata[note_type] = term_note.text

                    cursor.execute("""
                        INSERT INTO terms (concept_id, language, term, part_of_speech, term_metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (concept_id, lang, term_text, part_of_speech, json.dumps(term_metadata)))

        self.conn.commit()

    def load_csv(self, file_path, delimiter=','):
        """Loads a bilingual CSV file into the database."""
        cursor = self.conn.cursor()
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            std_fields = ['sourceterm', 'targetterm', 'definition', 'pos', 'source language code', 'target language code']
            
            for row_idx, row in enumerate(reader):
                src_term = row.get('sourceterm', '').strip()
                tgt_term = row.get('targetterm', '').strip()
                definition = row.get('definition', '').strip()
                pos = row.get('pos', '').strip() or None
                
                # Standardize database language tags cleanly (handles codes like cat, eng, ca-ES, etc.)
                src_lang = self._normalize_language_code(row.get('source language code', ''), for_spacy=False)
                tgt_lang = self._normalize_language_code(row.get('target language code', ''), for_spacy=False)
                
                if not src_term or not tgt_term or not src_lang or not tgt_lang:
                    continue
                
                concept_id = f"csv_concept_{row_idx}"
                definitions_dict = {"global": definition} if definition else {}
                
                cursor.execute("""
                    INSERT OR REPLACE INTO concepts (id, definitions, global_metadata)
                    VALUES (?, ?, ?)
                """, (concept_id, json.dumps(definitions_dict), json.dumps({})))
                
                extra_metadata = {k: v for k, v in row.items() if k not in std_fields and k is not None}
                extra_metadata_json = json.dumps(extra_metadata)
                
                cursor.execute("""
                    INSERT INTO terms (concept_id, language, term, part_of_speech, term_metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (concept_id, src_lang, src_term, pos, extra_metadata_json))
                
                cursor.execute("""
                    INSERT INTO terms (concept_id, language, term, part_of_speech, term_metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (concept_id, tgt_lang, tgt_term, pos, extra_metadata_json))
                
        self.conn.commit()

    def _get_terms_by_language(self, language):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT t.term, t.concept_id, t.part_of_speech, t.term_metadata, c.definitions, c.global_metadata
            FROM terms t
            JOIN concepts c ON t.concept_id = c.id
            WHERE t.language = ?
        """, (language,))
        return cursor.fetchall()

    def _get_translations(self, concept_id, source_language):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT language, term FROM terms 
            WHERE concept_id = ? AND language != ?
        """, (concept_id, source_language))
        return {row[0]: row[1] for row in cursor.fetchall()}

    def search(self, text, source_language, similarity_threshold=80):
        """
        Scans an input text using a greedy longest-match-first strategy.
        Uses standardized codes via langcodes, tokenizes via spaCy, and filters out 
        punctuation tokens so they don't accidentally consume valid terms.
        """
        # Standardize the incoming search string for precise database targeting
        db_lang = self._normalize_language_code(source_language, for_spacy=False)
        db_records = self._get_terms_by_language(db_lang)
        if not db_records:
            return []
            
        db_term_list = [record[0] for record in db_records]
        results = []
        
        # 1. Tokenize using your spaCy wrapper
        tokenizer = self._get_tokenizer(source_language)
        tokenized_string = tokenizer.tokenize(text, mode="tokenize")
        raw_tokens = tokenized_string.split()
        
        # 2. NEW: Filter out punctuation tokens (e.g., ':', ',', '.', '!', '?')
        # We only keep tokens that contain at least one alphanumeric character
        tokens = [t for t in raw_tokens if any(char.isalnum() for char in t)]
        num_tokens = len(tokens)
        
        consumed_indexes = [False] * num_tokens
        
        # Core Greedy Approach: sliding window from max_ngram down to 1
        for size in range(self.max_ngram, 0, -1):
            for i in range(num_tokens - size + 1):
                if any(consumed_indexes[i + k] for k in range(size)):
                    continue
                    
                candidate = " ".join(tokens[i:i+size])
                match = process.extractOne(candidate, db_term_list, scorer=fuzz.WRatio)
                
                if match and match[1] >= similarity_threshold:
                    matched_term = match[0]
                    score = match[1]
                    
                    for record in db_records:
                        if record[0] == matched_term:
                            # Block sub-tokens from being extracted individually later
                            for k in range(size):
                                consumed_indexes[i + k] = True
                                
                            translations = self._get_translations(record[1], db_lang)
                            
                            results.append({
                                "detected_text": candidate,
                                "matched_db_term": record[0],
                                "similarity_score": score,
                                "concept_id": record[1],
                                "part_of_speech": record[2],
                                "term_metadata": json.loads(record[3]),
                                "definitions": json.loads(record[4]),
                                "global_metadata": json.loads(record[5]),
                                "translations": translations
                            })
                            break 
                            
        return results

    def close(self):
        if self.conn:
            self.conn.close()
