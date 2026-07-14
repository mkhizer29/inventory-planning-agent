-- promotions_catalog : catalog price rules (automatic rule-based discounts)
-- mapped to products. Promotion calendar for spike attribution (FR-C3/C4).
SELECT
    cr.rule_id,
    cr.name,
    cr.from_date,
    cr.to_date,
    cr.is_active,
    cr.simple_action,
    cr.discount_amount,
    crp.product_id
FROM catalogrule cr
LEFT JOIN catalogrule_product crp ON crp.rule_id = cr.rule_id
;
