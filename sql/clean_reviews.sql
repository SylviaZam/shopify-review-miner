-- Strip punctuation, cast timestamp, lower‑case comment
CREATE OR REPLACE TABLE `${BQ_DATASET}.clean_${brand}` AS
SELECT
  brand,
  SAFE_CAST(created_at AS TIMESTAMP) AS ts,
  REGEXP_REPLACE(LOWER(comment), r'[^\\w\\s]', '') AS comment,
  question,
  response_id
FROM `${BQ_DATASET}.raw_${brand}`
WHERE comment IS NOT NULL;
