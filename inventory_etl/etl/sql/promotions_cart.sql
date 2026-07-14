-- promotions_cart : cart price rules / coupons (salesrule + coupon codes).
-- Promotion calendar part 2 — coupon-driven demand lift.
SELECT
    sr.rule_id,
    sr.name,
    sr.from_date,
    sr.to_date,
    sr.is_active,
    sr.simple_action,
    sr.discount_amount,
    sr.coupon_type,
    src.code AS coupon_code
FROM salesrule sr
LEFT JOIN salesrule_coupon src ON src.rule_id = sr.rule_id
;
