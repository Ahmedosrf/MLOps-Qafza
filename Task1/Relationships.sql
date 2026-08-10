-- ============================================================
-- OLIST DATABASE
-- COMPLETE DATABASE RELATIONSHIPS
-- PostgreSQL
-- ============================================================


-- ============================================================
-- 1. REMOVE OLD CONSTRAINTS
-- ============================================================

ALTER TABLE IF EXISTS olist_orders
DROP CONSTRAINT IF EXISTS fk_orders_customer CASCADE;

ALTER TABLE IF EXISTS olist_order_items
DROP CONSTRAINT IF EXISTS fk_order_items_order CASCADE;

ALTER TABLE IF EXISTS olist_order_items
DROP CONSTRAINT IF EXISTS fk_order_items_product CASCADE;

ALTER TABLE IF EXISTS olist_order_items
DROP CONSTRAINT IF EXISTS fk_order_items_seller CASCADE;

ALTER TABLE IF EXISTS olist_order_payments
DROP CONSTRAINT IF EXISTS fk_order_payments_order CASCADE;

ALTER TABLE IF EXISTS olist_order_reviews
DROP CONSTRAINT IF EXISTS fk_order_reviews_order CASCADE;

ALTER TABLE IF EXISTS olist_products
DROP CONSTRAINT IF EXISTS fk_products_category CASCADE;


-- Remove old Primary Keys if they were created before

ALTER TABLE IF EXISTS olist_customers
DROP CONSTRAINT IF EXISTS pk_olist_customers CASCADE;

ALTER TABLE IF EXISTS olist_orders
DROP CONSTRAINT IF EXISTS pk_olist_orders CASCADE;

ALTER TABLE IF EXISTS olist_products
DROP CONSTRAINT IF EXISTS pk_olist_products CASCADE;

ALTER TABLE IF EXISTS olist_sellers
DROP CONSTRAINT IF EXISTS pk_olist_sellers CASCADE;

ALTER TABLE IF EXISTS olist_order_items
DROP CONSTRAINT IF EXISTS pk_olist_order_items CASCADE;

ALTER TABLE IF EXISTS olist_order_payments
DROP CONSTRAINT IF EXISTS pk_olist_order_payments CASCADE;


-- Remove the category UNIQUE constraint if it was created
ALTER TABLE IF EXISTS product_category_name_translation
DROP CONSTRAINT IF EXISTS uq_product_category_name CASCADE;


-- ============================================================
-- 2. CHECK PRIMARY KEY DATA
-- ============================================================

SELECT
    'olist_customers' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customer_id) AS unique_ids,
    COUNT(*) FILTER (WHERE customer_id IS NULL) AS null_ids
FROM olist_customers

UNION ALL

SELECT
    'olist_orders',
    COUNT(*),
    COUNT(DISTINCT order_id),
    COUNT(*) FILTER (WHERE order_id IS NULL)
FROM olist_orders

UNION ALL

SELECT
    'olist_products',
    COUNT(*),
    COUNT(DISTINCT product_id),
    COUNT(*) FILTER (WHERE product_id IS NULL)
FROM olist_products

UNION ALL

SELECT
    'olist_sellers',
    COUNT(*),
    COUNT(DISTINCT seller_id),
    COUNT(*) FILTER (WHERE seller_id IS NULL)
FROM olist_sellers

UNION ALL

SELECT
    'olist_order_items (order_id, order_item_id)',
    COUNT(*),
    COUNT(DISTINCT (order_id, order_item_id)),
    COUNT(*) FILTER (
        WHERE order_id IS NULL
        OR order_item_id IS NULL
    )
FROM olist_order_items

UNION ALL

SELECT
    'olist_order_payments (order_id, payment_sequential)',
    COUNT(*),
    COUNT(DISTINCT (order_id, payment_sequential)),
    COUNT(*) FILTER (
        WHERE order_id IS NULL
        OR payment_sequential IS NULL
    )
FROM olist_order_payments;


-- ============================================================
-- 3. CREATE PRIMARY KEYS
-- ============================================================

ALTER TABLE olist_customers
ADD CONSTRAINT pk_olist_customers
PRIMARY KEY (customer_id);


ALTER TABLE olist_orders
ADD CONSTRAINT pk_olist_orders
PRIMARY KEY (order_id);


ALTER TABLE olist_products
ADD CONSTRAINT pk_olist_products
PRIMARY KEY (product_id);


ALTER TABLE olist_sellers
ADD CONSTRAINT pk_olist_sellers
PRIMARY KEY (seller_id);


-- Composite Primary Key
ALTER TABLE olist_order_items
ADD CONSTRAINT pk_olist_order_items
PRIMARY KEY (order_id, order_item_id);


-- Composite Primary Key
ALTER TABLE olist_order_payments
ADD CONSTRAINT pk_olist_order_payments
PRIMARY KEY (order_id, payment_sequential);


-- ============================================================
-- 4. CREATE FOREIGN KEYS
-- ============================================================


-- ------------------------------------------------------------
-- Customers → Orders
-- ------------------------------------------------------------

ALTER TABLE olist_orders
ADD CONSTRAINT fk_orders_customer
FOREIGN KEY (customer_id)
REFERENCES olist_customers(customer_id);


-- ------------------------------------------------------------
-- Orders → Order Items
-- ------------------------------------------------------------

ALTER TABLE olist_order_items
ADD CONSTRAINT fk_order_items_order
FOREIGN KEY (order_id)
REFERENCES olist_orders(order_id);


-- ------------------------------------------------------------
-- Products → Order Items
-- ------------------------------------------------------------

ALTER TABLE olist_order_items
ADD CONSTRAINT fk_order_items_product
FOREIGN KEY (product_id)
REFERENCES olist_products(product_id);


-- ------------------------------------------------------------
-- Sellers → Order Items
-- ------------------------------------------------------------

ALTER TABLE olist_order_items
ADD CONSTRAINT fk_order_items_seller
FOREIGN KEY (seller_id)
REFERENCES olist_sellers(seller_id);


-- ------------------------------------------------------------
-- Orders → Payments
-- ------------------------------------------------------------

ALTER TABLE olist_order_payments
ADD CONSTRAINT fk_order_payments_order
FOREIGN KEY (order_id)
REFERENCES olist_orders(order_id);


-- ------------------------------------------------------------
-- Orders → Reviews
-- ------------------------------------------------------------

ALTER TABLE olist_order_reviews
ADD CONSTRAINT fk_order_reviews_order
FOREIGN KEY (order_id)
REFERENCES olist_orders(order_id);


-- ============================================================
-- 5. DO NOT CREATE FK BETWEEN PRODUCTS AND CATEGORY TRANSLATION
-- ============================================================
--
-- We intentionally do NOT create:
--
-- olist_products.product_category_name
--          ↓
-- product_category_name_translation.product_category_name
--
-- because the translation table does not contain all
-- product categories found in olist_products.
--
-- Example:
-- pc_gamer exists in olist_products
-- but does not exist in product_category_name_translation.
--
-- Therefore this FK would violate referential integrity.
-- ============================================================


-- ============================================================
-- 6. VERIFY PRIMARY KEYS
-- ============================================================

SELECT
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type
FROM information_schema.table_constraints tc
WHERE tc.table_schema = 'public'
AND tc.constraint_type = 'PRIMARY KEY'
ORDER BY tc.table_name;


-- ============================================================
-- 7. VERIFY FOREIGN KEYS
-- ============================================================

SELECT
    tc.table_name,
    tc.constraint_name,
    kcu.column_name,
    ccu.table_name AS referenced_table,
    ccu.column_name AS referenced_column
FROM information_schema.table_constraints AS tc

JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema

JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema

WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public'

ORDER BY
    tc.table_name,
    tc.constraint_name;


-- ============================================================
-- 8. FINAL CHECK
-- ============================================================

SELECT
    'Database relationships created successfully!' AS status;


SELECT
    tc.table_name AS child_table,
    kcu.column_name AS child_column,
    ccu.table_name AS parent_table,
    ccu.column_name AS parent_column,
    tc.constraint_name
FROM information_schema.table_constraints AS tc

JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema

JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema

WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public'

ORDER BY tc.table_name;