-- iwashere schema. Applied by server.db.conn.init_db().
-- Verified against SQLite 3.53.1 + sqlite-vec v0.1.9 (M0).

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS items (
  id            INTEGER PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('note','image')),
  status        TEXT NOT NULL DEFAULT 'processing'
                  CHECK (status IN ('processing','ready','error')),
  title         TEXT,
  body          TEXT,
  blob_path     TEXT,
  thumb_path    TEXT,
  mime          TEXT,
  width         INTEGER,
  height        INTEGER,
  bytes         INTEGER,
  content_hash  TEXT UNIQUE,
  dhash         BLOB,
  near_dup_of   INTEGER REFERENCES items(id) ON DELETE SET NULL,
  project_id    INTEGER REFERENCES projects(id) ON DELETE SET NULL,
  caption       TEXT,
  caption_model TEXT,
  layout_text   TEXT,
  flow_text     TEXT,
  boxes_json    TEXT,
  source        TEXT NOT NULL CHECK (source IN ('watch','upload','paste','compose')),
  source_path   TEXT,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  seen_at       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_project ON items(project_id);
CREATE INDEX IF NOT EXISTS idx_items_status  ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_dup     ON items(near_dup_of);

CREATE TABLE IF NOT EXISTS projects (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  description TEXT,
  color       TEXT,
  created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS vocab_terms (
  id              INTEGER PRIMARY KEY,
  term            TEXT NOT NULL UNIQUE,
  kind            TEXT,
  joytag_alias    TEXT,
  -- Some concepts are only meaningful on an image that contains text.
  -- CLIP will happily rank a photograph as the most "whiteboard-like"
  -- thing in a corpus of photographs; requiring OCR evidence kills that
  -- whole class of false positive using information already on hand.
  requires_text   INTEGER NOT NULL DEFAULT 0,
  prompt_template TEXT NOT NULL DEFAULT 'a photo of {term}',
  threshold       REAL NOT NULL DEFAULT 0.40,
  clip_mu         REAL,
  clip_sigma      REAL,
  state           TEXT NOT NULL DEFAULT 'active'
                    CHECK (state IN ('active','pending','merged','retired')),
  merged_into     INTEGER REFERENCES vocab_terms(id),
  accept_count    INTEGER NOT NULL DEFAULT 0,
  reject_count    INTEGER NOT NULL DEFAULT 0,
  created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vocab_state ON vocab_terms(state);

CREATE TABLE IF NOT EXISTS vocab_aliases (
  term_id INTEGER NOT NULL REFERENCES vocab_terms(id) ON DELETE CASCADE,
  alias   TEXT NOT NULL,
  PRIMARY KEY (alias)
);

CREATE TABLE IF NOT EXISTS item_tags (
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  term_id    INTEGER NOT NULL REFERENCES vocab_terms(id) ON DELETE CASCADE,
  confidence REAL NOT NULL,
  source     TEXT NOT NULL CHECK (source IN ('human','machine')),
  state      TEXT NOT NULL CHECK (state IN ('accepted','pending','rejected')),
  created_at INTEGER NOT NULL,
  PRIMARY KEY (item_id, term_id)
);
CREATE INDEX IF NOT EXISTS idx_item_tags_term ON item_tags(term_id, state);

CREATE TABLE IF NOT EXISTS tag_scores (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  term_id INTEGER NOT NULL REFERENCES vocab_terms(id) ON DELETE CASCADE,
  joytag  REAL,
  clip    REAL,
  vlm     REAL,
  PRIMARY KEY (item_id, term_id)
);

CREATE TABLE IF NOT EXISTS links (
  id         INTEGER PRIMARY KEY,
  src        INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  dst        INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL CHECK (kind IN ('manual','semantic')),
  weight     REAL NOT NULL DEFAULT 1.0,
  label      TEXT,
  created_at INTEGER NOT NULL,
  UNIQUE (src, dst, kind),
  CHECK (src <> dst)
);
CREATE INDEX IF NOT EXISTS idx_links_src ON links(src, kind);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst, kind);

CREATE TABLE IF NOT EXISTS entities (
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,
  value      TEXT NOT NULL,
  confidence REAL NOT NULL,
  PRIMARY KEY (item_id, kind, value)
);
CREATE INDEX IF NOT EXISTS idx_entities_kv ON entities(kind, value);

CREATE TABLE IF NOT EXISTS todos (
  id         INTEGER PRIMARY KEY,
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  text       TEXT NOT NULL,
  due_date   TEXT,
  done       INTEGER NOT NULL DEFAULT 0,
  confidence REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_queue (
  id          INTEGER PRIMARY KEY,
  item_id     INTEGER REFERENCES items(id) ON DELETE CASCADE,
  stage       TEXT NOT NULL,
  attempts    INTEGER NOT NULL DEFAULT 0,
  escalate    INTEGER NOT NULL DEFAULT 0,
  enqueued_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queue_stage ON ingest_queue(stage, enqueued_at);

CREATE TABLE IF NOT EXISTS item_errors (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  stage   TEXT NOT NULL,
  error   TEXT NOT NULL,
  ts      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
  id      INTEGER PRIMARY KEY,
  ts      INTEGER NOT NULL,
  action  TEXT NOT NULL,
  payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Contentless FTS5. Written explicitly in the finalize stage, NOT by triggers:
-- triggers would fire mid-enrichment while flow_text and caption are still null.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_items USING fts5(
  title, body, flow_text, caption, tags,
  content='',
  tokenize='unicode61 remove_diacritics 2'
);

-- Vector spaces. Never mixed: one table per embedding space.
-- Qwen3-Embedding-0.6B, 1024-d. Notes always; images too when
-- embeddings.qwen_covers_image_text is on (spec 9.4).
CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0(
  item_id INTEGER PRIMARY KEY,
  embedding FLOAT[1024]
);

-- jina-clip-v2 shared space.
--   kind='image'   -> visual tower over the image
--   kind='imgtext' -> text tower over caption + flow_text
CREATE VIRTUAL TABLE IF NOT EXISTS vec_jina USING vec0(
  rowid INTEGER PRIMARY KEY,
  item_id INTEGER,
  kind TEXT,
  embedding FLOAT[1024]
);

-- Vocabulary term embeddings (Qwen3-Embedding-0.6B) for the merge gate
-- and for tag-seed expansion at query time.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_vocab USING vec0(
  term_id INTEGER PRIMARY KEY,
  embedding FLOAT[1024]
);

-- Perceptual hashes for near-duplicate detection.
-- Verified M0: the declaration is bit[64] (binary[64] is rejected) and
-- inserts/queries must wrap the blob in vec_bit(). Distance is Hamming.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_phash USING vec0(
  item_id INTEGER PRIMARY KEY,
  hash BIT[64]
);
