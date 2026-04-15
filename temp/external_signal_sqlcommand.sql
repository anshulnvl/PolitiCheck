CREATE TABLE IF NOT EXISTS fact_check_cache (
    id SERIAL PRIMARY KEY,
    domain TEXT UNIQUE NOT NULL,
    newsguard_score INTEGER,           -- 0–100
    mbfc_rating TEXT,                  -- "HIGH", "MIXED", "LOW", "SATIRE"
    domain_age_days INTEGER,
    alexa_rank INTEGER,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Seed with known sources (expand this list)
INSERT INTO fact_check_cache (domain, newsguard_score, mbfc_rating) VALUES
    ('reuters.com',      100, 'HIGH'),
    ('apnews.com',       100, 'HIGH'),
    ('bbc.com',           94, 'HIGH'),
    ('naturalnews.com',    0, 'LOW'),
    ('infowars.com',       0, 'LOW'),
    ('theonion.com',      NULL, 'SATIRE')
ON CONFLICT (domain) DO NOTHING;