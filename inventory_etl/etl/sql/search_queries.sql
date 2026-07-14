-- search_queries : on-site search volume + zero-result searches.
-- High-value cold-start signal: most of the catalog has never sold, and a
-- search term may be the only demand signal that exists for those SKUs.
-- {TABLE} is resolved by extract.py (search_query or search_query_1).
SELECT
    query_text,
    num_results,
    popularity,
    updated_at
FROM {TABLE}
;
