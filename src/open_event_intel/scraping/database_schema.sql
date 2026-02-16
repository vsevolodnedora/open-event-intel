CREATE TABLE IF NOT EXISTS "{table_name}" (
    ID TEXT PRIMARY KEY,
    published_on TIMESTAMP NOT NULL,
    title TEXT NOT NULL,
    added_on TIMESTAMP NOT NULL,
    url TEXT NOT NULL,
    language TEXT NOT NULL,
    post BLOB NOT NULL
);