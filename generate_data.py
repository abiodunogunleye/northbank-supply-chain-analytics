#!/usr/bin/env python3
"""Generate the FICTIONAL Northbank Logistics supply chain dataset.

Northbank Logistics is an invented UK warehouse business. Nothing here is real.

Usage:
    python generate_data.py

Writes 7 CSV files into ./data/ covering the 90 days ending 15 November 2026.
The output is identical on every run (fixed random seed).

How it works
------------
1. Build static tables (suppliers, products, customers).
2. Generate all customer demand (orders + lines) up front.
3. Simulate the warehouse day by day: receive POs, dispatch orders first-in
   first-out within labour capacity, track stock, raise POs when stock is low.
4. Derive every output table from that one simulation, so orders, lines, stock,
   POs and labour all reconcile.
5. validate() checks the clean data reconciles.
6. inject_defects() then plants the data-quality problems into orders.csv.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "data"

N_DAYS = 90
START = pd.Timestamp("2026-08-18")
END = START + pd.Timedelta(days=N_DAYS - 1)  # 2026-11-15
DATES = pd.date_range(START, periods=N_DAYS)

# Busy period (autumn promotion): demand surges faster than capacity.
PEAK_START = pd.Timestamp("2026-10-12")
PEAK_END = pd.Timestamp("2026-10-25")
PEAK_DEMAND_MULT = 1.7
PEAK_OVERTIME_MULT = 1.25  # longer shifts in the peak
NIGHT_PEAK_PRODUCTIVITY = 0.75  # planted: Night shift picks 25% slower on peak days

TARGET_ORDERS = 6000
WEEKEND_FACTOR = 0.6
MAX_LINES = 8
CANCEL_RATE = 0.01
SAME_DAY_ELIGIBLE = 0.6  # share of orders placed before the daily cut-off

N_SKUS = 300
SKUS_PER_CATEGORY = 50
N_CUSTOMERS = 800

SHIFT_HOURS = 8.0
UPH_DAY = 24.0  # units picked & packed per operative-hour
UPH_NIGHT = 23.0

RP_SAFETY = 2.2  # reorder point = demand * (RP_SAFETY * lead time + RP_FLAT) days
RP_FLAT = 6
LUMP_BUFFER = 60  # units of safety stock for big trade orders
LATE_SUPPLIER_BUFFER_DAYS = 14
PO_COVER_DAYS = 55
PACK = 6  # POs are rounded up to whole case packs

N_STOCKOUT_SKUS = 12  # planted: SKUs deliberately under-stocked
N_STOCKOUT_FROM_LATE = 9

DUPLICATE_RATE = 0.02
REGION_MISSPELL_RATE = 0.06
BLANK_DISPATCHED, BLANK_DELIVERED, BLANK_PROMISED = 15, 15, 5

# --------------------------------------------------------------------------
# Reference data (all invented)
# --------------------------------------------------------------------------
SUPPLIERS = [
    ("SUP-01", "Harrowgate Home Supplies Ltd", "United Kingdom"),
    ("SUP-02", "Calder Valley Housewares Ltd", "United Kingdom"),
    ("SUP-03", "Ironbridge Tools & Fixings Ltd", "United Kingdom"),
    ("SUP-04", "Brenner Werkzeug GmbH", "Germany"),
    ("SUP-05", "Greenfield Garden Products Ltd", "United Kingdom"),
    ("SUP-06", "Tuin & Co BV", "Netherlands"),
    ("SUP-07", "Pawprint Pet Supplies Ltd", "United Kingdom"),
    ("SUP-08", "Nowak Zoo Sp. z o.o.", "Poland"),
    ("SUP-09", "Shenlan Electronics Co Ltd", "China"),
    ("SUP-10", "Mekong Accessories JSC", "Vietnam"),
    ("SUP-11", "Bellezza Cosmetici SRL", "Italy"),
    ("SUP-12", "Kerala Wellness Exports Pvt Ltd", "India"),
]
LATE_SUPPLIERS = {"SUP-04", "SUP-09"}  # planted: chronically late
COUNTRY_LEAD_DAYS = {
    "United Kingdom": 4, "Germany": 8, "Netherlands": 8, "Poland": 9,
    "Italy": 9, "China": 25, "Vietnam": 20, "India": 21,
}
CATEGORY_SUPPLIERS = {
    "Homeware": ["SUP-01", "SUP-02"],
    "DIY & Tools": ["SUP-03", "SUP-04"],
    "Garden": ["SUP-05", "SUP-06"],
    "Pet Supplies": ["SUP-07", "SUP-08"],
    "Electronics Accessories": ["SUP-09", "SUP-10"],
    "Health & Beauty": ["SUP-11", "SUP-12"],
}
COST_MEDIAN = {
    "Homeware": 6.0, "DIY & Tools": 9.0, "Garden": 7.0,
    "Pet Supplies": 4.5, "Electronics Accessories": 8.0, "Health & Beauty": 3.5,
}
VOCAB = {  # category -> (adjectives, nouns, variants)
    "Homeware": (
        ["Stainless Steel", "Ceramic", "Bamboo", "Glass", "Cotton", "Non-Stick"],
        ["Saucepan", "Storage Jar", "Chopping Board", "Tea Towel Set", "Mug",
         "Laundry Basket", "Cutlery Set", "Baking Tray", "Food Container", "Kettle"],
        ["Small", "Medium", "Large", "Set of 4"]),
    "DIY & Tools": (
        ["Heavy Duty", "Cordless", "Galvanised", "Pro", "Compact", "Rust-Resistant"],
        ["Claw Hammer", "Screwdriver Set", "Spirit Level", "Tape Measure", "Wood Screws",
         "Cable Ties", "Adjustable Spanner", "Utility Knife", "Drill Bit Set", "Wall Plugs"],
        ["Standard", "Pack of 50", "Pack of 100", "XL"]),
    "Garden": (
        ["Weatherproof", "UV-Resistant", "Organic", "Heavy Duty", "Folding", "Lightweight"],
        ["Plant Pot", "Garden Hose", "Compost Bag", "Hand Trowel", "Pruning Shears",
         "Watering Can", "Bird Feeder", "Garden Gloves", "Weed Membrane", "Seed Tray"],
        ["Small", "Medium", "Large", "Multipack"]),
    "Pet Supplies": (
        ["Grain-Free", "Chew-Resistant", "Natural", "Premium", "Washable", "Puppy"],
        ["Dog Lead", "Cat Litter", "Dry Dog Food", "Cat Scratching Post", "Pet Bowl",
         "Dog Chew Toy", "Cat Treats", "Pet Bed", "Dog Shampoo", "Pet Carrier"],
        ["Small", "Medium", "Large", "Value Pack"]),
    "Electronics Accessories": (
        ["Braided", "Fast-Charge", "Universal", "Compact", "Premium", "Slim"],
        ["USB-C Cable", "Phone Case", "Screen Protector", "Wall Charger", "HDMI Cable",
         "Wireless Mouse", "Laptop Sleeve", "Power Bank", "Earphones", "Car Phone Holder"],
        ["Black", "White", "1m", "2m"]),
    "Health & Beauty": (
        ["Gentle", "Fragrance-Free", "Nourishing", "Daily", "Intensive", "Travel"],
        ["Hand Cream", "Shampoo", "Body Wash", "Sunscreen SPF30", "Lip Balm",
         "Face Wash", "Conditioner", "Deodorant", "Hand Sanitiser", "Shower Gel"],
        ["100ml", "250ml", "500ml", "Twin Pack"]),
}

REGIONS = ["London", "South East", "South West", "Midlands", "North West",
           "North East", "Yorkshire", "Scotland", "Wales"]
REGION_WEIGHTS = [0.20, 0.17, 0.08, 0.16, 0.12, 0.05, 0.09, 0.07, 0.06]
PROMISE_DAYS = {"London": 2, "South East": 2, "South West": 3, "Midlands": 3,
                "North West": 3, "North East": 4, "Yorkshire": 3, "Scotland": 4, "Wales": 3}
TRANSIT_DAYS = {"London": (1, 1), "South East": (1, 1), "South West": (1, 2),
                "Midlands": (1, 2), "North West": (1, 2), "North East": (1, 2),
                "Yorkshire": (1, 2), "Scotland": (2, 3), "Wales": (1, 2)}
REGION_VARIANTS = {  # planted: messy spellings
    "London": ["london", "LONDON", "Londn", "London "],
    "South East": ["SE", "S.East", "south east", "South-East", "South East "],
    "South West": ["SW", "S. West", "south west", "South-West"],
    "Midlands": ["midlands", "Mid lands", "MIDLANDS", "Midland"],
    "North West": ["N. West", "North-West", "north west", "NW"],
    "North East": ["N. East", "North-East", "north east", "NE"],
    "Yorkshire": ["yorkshire", "Yorks", "YORKSHIRE", "Yorkshire & Humber"],
    "Scotland": ["scotland", "SCOTLAND", "Scotland ", "Scot"],
    "Wales": ["wales", "WALES", "Wales ", "Cymru"],
}

QTY_VALUES = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48])
QTY_WEIGHTS = np.array([10, 10, 10, 9, 8, 8, 8, 7, 6, 5, 5, 4, 3, 2], dtype=float)
QTY_WEIGHTS /= QTY_WEIGHTS.sum()


def supplier_behaviour(supplier_id: str) -> dict:
    if supplier_id in LATE_SUPPLIERS:
        return {"p_late": 0.65, "delay": (3, 12), "p_short": 0.35}
    return {"p_late": 0.08, "delay": (1, 3), "p_short": 0.06}


# --------------------------------------------------------------------------
# Step 1: static tables
# --------------------------------------------------------------------------
def build_suppliers() -> pd.DataFrame:
    return pd.DataFrame(SUPPLIERS, columns=["supplier_id", "name", "country"])


def build_products(rng: np.random.Generator, suppliers: pd.DataFrame) -> pd.DataFrame:
    country = dict(zip(suppliers.supplier_id, suppliers.country))
    rows = []
    for cat, (adjs, nouns, variants) in VOCAB.items():
        combos = [(a, n, v) for a in adjs for n in nouns for v in variants]
        for k in rng.choice(len(combos), size=SKUS_PER_CATEGORY, replace=False):
            a, n, v = combos[k]
            sup = str(rng.choice(CATEGORY_SUPPLIERS[cat]))
            cost = round(float(np.clip(rng.lognormal(np.log(COST_MEDIAN[cat]), 0.5), 0.8, 80)), 2)
            price = round(cost * float(rng.uniform(1.3, 2.2)), 2)
            lead = COUNTRY_LEAD_DAYS[country[sup]] + int(rng.integers(-1, 2))
            rows.append({"description": f"{a} {n} - {v}", "category": cat,
                         "unit_cost": cost, "unit_price": price,
                         "supplier_id": sup, "lead_time_days": lead})
    df = pd.DataFrame(rows)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.insert(0, "sku", [f"NB-{10001 + i}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------
# Step 2: demand
# --------------------------------------------------------------------------
def generate_demand(rng, pop, cust_p, cust_region):
    is_peak = ((DATES >= PEAK_START) & (DATES <= PEAK_END))
    weights = np.array([(WEEKEND_FACTOR if d.weekday() >= 5 else 1.0) for d in DATES])
    weights = weights * np.where(is_peak, PEAK_DEMAND_MULT, 1.0)
    counts = rng.poisson(TARGET_ORDERS * weights / weights.sum())

    orders = []
    for day, n in enumerate(counts):
        for _ in range(int(n)):
            n_lines = min(MAX_LINES, 1 + int(rng.poisson(2.4)))
            skus = rng.choice(N_SKUS, size=n_lines, replace=False, p=pop)
            qtys = rng.choice(QTY_VALUES, size=n_lines, p=QTY_WEIGHTS)
            cust = int(rng.choice(N_CUSTOMERS, p=cust_p))
            orders.append({
                "day": day,
                "customer": cust,
                "region": cust_region[cust],
                "lines": [(int(s), int(q)) for s, q in zip(skus, qtys)],
                "cancelled": bool(rng.random() < CANCEL_RATE),
                "eligible": day if rng.random() < SAME_DAY_ELIGIBLE else day + 1,
                "dispatch_day": None, "shipped": None, "delivered_day": None,
            })
    for i, o in enumerate(orders):
        o["order_id"] = f"ORD-{i + 1:06d}"
    return orders, is_peak


# --------------------------------------------------------------------------
# Step 3: day-by-day warehouse simulation
# --------------------------------------------------------------------------
def simulate(rng, orders, products, is_peak, d, rp, init_stock, is_target):
    n = len(products)
    supplier = products["supplier_id"].to_numpy()
    lead = products["lead_time_days"].to_numpy()

    on_hand = init_stock.astype(np.int64).copy()
    open_demand = np.zeros(n, dtype=np.int64)
    on_order = np.zeros(n, dtype=np.int64)
    po_open = np.zeros(n, dtype=bool)
    receipts = defaultdict(list)
    pos = []
    queue = []
    by_day = defaultdict(list)
    for o in orders:
        by_day[o["day"]].append(o)

    inv_on_hand = np.zeros((N_DAYS, n), dtype=np.int64)
    inv_alloc = np.zeros((N_DAYS, n), dtype=np.int64)
    labour = []
    backlog = []

    for day in range(N_DAYS):
        # 1. purchase orders arriving today
        for po in receipts.pop(day, []):
            on_hand[po["sku_idx"]] += po["qty_received"]
            on_order[po["sku_idx"]] -= po["qty_ordered"]
            po_open[po["sku_idx"]] = False

        # 2. new orders join the queue
        for o in by_day[day]:
            if o["cancelled"]:
                continue
            queue.append(o)
            for s, q in o["lines"]:
                open_demand[s] += q

        # 3. today's labour capacity
        weekend = DATES[day].weekday() >= 5
        peak = bool(is_peak[day])
        ops_d = int(rng.integers(4, 7)) if weekend else int(rng.integers(7, 10))
        ops_n = int(rng.integers(3, 5)) if weekend else int(rng.integers(5, 8))
        hrs = SHIFT_HOURS * (PEAK_OVERTIME_MULT if peak else 1.0)
        uph_d = UPH_DAY * float(rng.normal(1.0, 0.04))
        uph_n = UPH_NIGHT * float(rng.normal(1.0, 0.04))
        if peak:
            uph_n *= NIGHT_PEAK_PRODUCTIVITY
        cap_d, cap_n = ops_d * hrs * uph_d, ops_n * hrs * uph_n
        cap = cap_d + cap_n

        # 4. dispatch first-in-first-out until capacity is used
        used, n_disp, full, remaining = 0, 0, False, []
        for o in queue:
            if full or o["eligible"] > day:
                remaining.append(o)
                continue
            ship = [min(q, int(on_hand[s])) for s, q in o["lines"]]
            tot = sum(ship)
            if tot == 0:  # nothing pickable: order is held until stock arrives
                remaining.append(o)
                continue
            if used + tot > cap:
                full = True
                remaining.append(o)
                continue
            for (s, q), sh in zip(o["lines"], ship):
                on_hand[s] -= sh
                open_demand[s] -= q
            lo, hi = TRANSIT_DAYS[o["region"]]
            transit = int(rng.integers(lo, hi + 1))
            if rng.random() < 0.07:  # occasional carrier delay
                transit += int(rng.integers(1, 3))
            o["dispatch_day"], o["shipped"] = day, ship
            o["delivered_day"] = day + transit
            used += tot
            n_disp += 1
        queue = remaining
        backlog.append(len(queue))

        # 5. labour rows (hours flex with workload; units split by capacity)
        util = used / cap
        units_d = int(round(used * cap_d / cap))
        orders_d = int(round(n_disp * cap_d / cap))
        for shift, ops, units, n_ord in (
                ("Day", ops_d, units_d, orders_d),
                ("Night", ops_n, used - units_d, n_disp - orders_d)):
            labour.append({"date": DATES[day], "shift": shift, "operatives": ops,
                           "hours_worked": round(ops * hrs * min(util, 1.0), 1),
                           "units_picked": units, "orders_picked": n_ord})

        # 6. replenishment: raise a PO when stock position hits the reorder point
        position = on_hand + on_order - open_demand
        for s in np.nonzero((position <= rp) & ~po_open)[0]:
            cover = 12 if is_target[s] else PO_COVER_DAYS
            qty = max(int(math.ceil(d[s] * cover / PACK)) * PACK, 4 * PACK)
            beh = supplier_behaviour(supplier[s])
            delay = 0
            if rng.random() < beh["p_late"]:
                delay = int(rng.integers(beh["delay"][0], beh["delay"][1] + 1))
            received_qty = qty
            if rng.random() < beh["p_short"]:
                received_qty = int(math.ceil(qty * float(rng.uniform(0.7, 0.92))))
            promised = day + int(lead[s])
            arrive = promised + delay
            po = {"supplier_id": supplier[s], "sku_idx": int(s), "qty_ordered": qty,
                  "order_day": day, "promised_day": promised, "arrive_day": arrive,
                  "qty_received": received_qty}
            pos.append(po)
            on_order[s] += qty
            po_open[s] = True
            if arrive < N_DAYS:
                receipts[arrive].append(po)

        # 7. end-of-day stock snapshot
        inv_on_hand[day] = on_hand
        inv_alloc[day] = np.minimum(on_hand, open_demand)

    return {"orders": orders, "pos": pos, "inv_on_hand": inv_on_hand,
            "inv_alloc": inv_alloc, "labour": labour, "backlog": backlog}


# --------------------------------------------------------------------------
# Step 4: build output tables from the simulation
# --------------------------------------------------------------------------
def day_to_date(day):
    return DATES[day] if day is not None and day < N_DAYS else pd.NaT


def build_tables(sim, products, suppliers):
    sku_ids = products["sku"].to_numpy()
    order_rows, line_rows = [], []
    for o in sim["orders"]:
        shipped = o["shipped"] if o["shipped"] is not None else [0] * len(o["lines"])
        for (s, q), sh in zip(o["lines"], shipped):
            line_rows.append({"order_id": o["order_id"], "sku": sku_ids[s],
                              "qty_ordered": q, "qty_shipped": sh})
        dispatched = o["dispatch_day"] is not None
        delivered = dispatched and o["delivered_day"] < N_DAYS
        if o["cancelled"]:
            status = "Cancelled"
        elif delivered:
            status = "Delivered"
        elif dispatched:
            status = "In Transit"
        else:
            status = "Awaiting Dispatch"
        order_rows.append({
            "order_id": o["order_id"], "order_date": DATES[o["day"]],
            "customer_id": f"CUST-{o['customer'] + 1:04d}", "region": o["region"],
            "promised_date": DATES[o["day"]] + pd.Timedelta(days=PROMISE_DAYS[o["region"]]),
            "dispatched_date": day_to_date(o["dispatch_day"]),
            "delivered_date": day_to_date(o["delivered_day"]) if delivered else pd.NaT,
            "lines_ordered": len(o["lines"]),
            "lines_shipped": sum(1 for (s, q), sh in zip(o["lines"], shipped) if sh == q),
            "status": status})
    orders = pd.DataFrame(order_rows)
    lines = pd.DataFrame(line_rows)

    po_rows = []
    for i, po in enumerate(sim["pos"]):
        arrived = po["arrive_day"] < N_DAYS
        po_rows.append({
            "po_id": f"PO-{i + 1:05d}", "supplier_id": po["supplier_id"],
            "sku": sku_ids[po["sku_idx"]], "qty_ordered": po["qty_ordered"],
            "qty_received": po["qty_received"] if arrived else 0,
            "order_date": DATES[po["order_day"]],
            "promised_date": START + pd.Timedelta(days=po["promised_day"]),
            "received_date": DATES[po["arrive_day"]] if arrived else pd.NaT})
    pos = pd.DataFrame(po_rows)

    inventory = pd.DataFrame({
        "date": np.repeat(DATES, len(sku_ids)),
        "sku": np.tile(sku_ids, N_DAYS),
        "on_hand": sim["inv_on_hand"].ravel(),
        "allocated": sim["inv_alloc"].ravel()})
    labour = pd.DataFrame(sim["labour"])
    return orders, lines, inventory, pos, labour


# --------------------------------------------------------------------------
# Step 5: validation (runs on the CLEAN data, before defects are planted)
# --------------------------------------------------------------------------
def validate(orders, lines, products, inventory, suppliers, pos, labour):
    def check(cond, msg):
        if not bool(cond):
            raise AssertionError(f"VALIDATION FAILED: {msg}")

    n_sku = len(products)
    # row counts in expected ranges
    check(5800 <= len(orders) <= 6200, f"orders rows {len(orders)}")
    check(18000 <= len(lines) <= 22000, f"order_lines rows {len(lines)}")
    check(len(products) == 300, "products rows")
    check(len(inventory) == 300 * N_DAYS, "inventory rows")
    check(len(suppliers) == 12, "suppliers rows")
    check(540 <= len(pos) <= 660, f"purchase_orders rows {len(pos)}")
    check(len(labour) == 180, "labour rows")

    # keys
    check(set(lines.order_id) <= set(orders.order_id), "orphan order lines")
    check(orders.order_id.is_unique, "order_id not unique")
    check(set(products.supplier_id) <= set(suppliers.supplier_id), "unknown supplier on product")
    check(set(pos.sku) <= set(products.sku), "PO for unknown SKU")
    sup_of = dict(zip(products.sku, products.supplier_id))
    check((pos.sku.map(sup_of) == pos.supplier_id).all(), "PO supplier != product supplier")

    # lines vs orders
    check((lines.qty_shipped >= 0).all() and (lines.qty_shipped <= lines.qty_ordered).all(),
          "qty_shipped out of range")
    g = lines.groupby("order_id")
    n_lines = g.size()
    n_full = (lines.qty_shipped == lines.qty_ordered).groupby(lines.order_id).sum()
    o = orders.set_index("order_id")
    check((o.lines_ordered == n_lines.reindex(o.index)).all(), "lines_ordered mismatch")
    check((o.lines_shipped == n_full.reindex(o.index)).all(), "lines_shipped mismatch")
    shipped_units = lines.groupby("order_id").qty_shipped.sum().reindex(o.index)

    # dates and status
    disp = orders.dispatched_date.notna()
    check((orders.promised_date > orders.order_date).all(), "promised <= order date")
    check((orders.dispatched_date[disp] >= orders.order_date[disp]).all(), "dispatch before order")
    dlv = orders.delivered_date.notna()
    check((orders.delivered_date[dlv] >= orders.dispatched_date[dlv]).all(), "delivery before dispatch")
    check((orders.delivered_date[dlv] <= END).all(), "delivery after end date")
    st = orders.status
    check((st[st == "Delivered"].index.isin(orders.index[dlv])).all(), "Delivered w/o date")
    check(not dlv[st != "Delivered"].any(), "delivered_date on non-Delivered order")
    check(disp[st.isin(["Delivered", "In Transit"])].all(), "shipped order w/o dispatch date")
    check(not disp[st.isin(["Awaiting Dispatch", "Cancelled"])].any(), "unshipped order has dispatch")
    check((shipped_units[~disp.to_numpy()] == 0).all(), "undispatched order has shipped qty")
    check((shipped_units[disp.to_numpy()] > 0).all(), "dispatched order shipped nothing")

    # stock roll-forward: on_hand[t] = on_hand[t-1] + received[t] - shipped[t]
    sku_idx = {s: i for i, s in enumerate(products.sku)}
    check((inventory.sku.to_numpy().reshape(N_DAYS, n_sku)[0] == products.sku.to_numpy()).all(),
          "inventory SKU order")
    oh = inventory.on_hand.to_numpy().reshape(N_DAYS, n_sku)
    al = inventory.allocated.to_numpy().reshape(N_DAYS, n_sku)
    check((oh >= 0).all(), "negative on_hand")
    check((al >= 0).all() and (al <= oh).all(), "allocated out of range")
    rec = np.zeros_like(oh)
    r = pos[pos.received_date.notna()]
    np.add.at(rec, ((r.received_date - START).dt.days.to_numpy(),
                    r.sku.map(sku_idx).to_numpy()), r.qty_received.to_numpy())
    shp = np.zeros_like(oh)
    m = lines.merge(orders[["order_id", "dispatched_date"]], on="order_id")
    m = m[m.dispatched_date.notna()]
    np.add.at(shp, ((m.dispatched_date - START).dt.days.to_numpy(),
                    m.sku.map(sku_idx).to_numpy()), m.qty_shipped.to_numpy())
    check((oh[1:] == oh[:-1] + rec[1:] - shp[1:]).all(), "stock roll-forward broken")

    # PO sanity
    check((pos.promised_date > pos.order_date).all(), "PO promised <= order")
    rr = pos.received_date.notna()
    check((pos.received_date[rr] >= pos.order_date[rr]).all(), "PO received before ordered")
    check((pos.qty_received <= pos.qty_ordered).all(), "PO over-received")
    check((pos.qty_received[~rr] == 0).all(), "unreceived PO has qty_received")

    # labour reconciles with dispatches
    lab = labour.groupby("date")[["units_picked", "orders_picked"]].sum()
    units_by_day = pd.Series(shp.sum(axis=1), index=DATES)
    orders_by_day = orders.dispatched_date.value_counts().reindex(DATES, fill_value=0)
    check((lab.units_picked.reindex(DATES) == units_by_day).all(), "labour units != shipped units")
    check((lab.orders_picked.reindex(DATES) == orders_by_day).all(), "labour orders != dispatched orders")

    # planted stock-outs
    stockout_skus = int((oh.min(axis=0) == 0).sum())
    check(10 <= stockout_skus <= 15, f"{stockout_skus} SKUs stocked out (need 10-15)")
    return stockout_skus


# --------------------------------------------------------------------------
# Step 6: plant the data-quality problems (orders.csv only)
# --------------------------------------------------------------------------
def inject_defects(rng, orders):
    df = orders.copy()
    n = len(df)
    report = {}

    # mixed spellings of region names
    k = int(round(REGION_MISSPELL_RATE * n))
    region = df.region.to_numpy(dtype=object).copy()
    for i in rng.choice(n, size=k, replace=False):
        region[i] = str(rng.choice(REGION_VARIANTS[region[i]]))
    df["region"] = region
    report["region_respelled"] = k

    # a few blank dates (on rows that really had them)
    delivered_rows = np.nonzero((df.status == "Delivered").to_numpy())[0]
    picks = rng.choice(delivered_rows, size=BLANK_DISPATCHED + BLANK_DELIVERED, replace=False)
    df.iloc[picks[:BLANK_DISPATCHED], df.columns.get_loc("dispatched_date")] = pd.NaT
    df.iloc[picks[BLANK_DISPATCHED:], df.columns.get_loc("delivered_date")] = pd.NaT
    others = np.setdiff1d(np.arange(n), picks)
    df.iloc[rng.choice(others, size=BLANK_PROMISED, replace=False),
            df.columns.get_loc("promised_date")] = pd.NaT
    report["blank_dates"] = BLANK_DISPATCHED + BLANK_DELIVERED + BLANK_PROMISED

    # exact duplicate orders (double-loaded rows)
    m = int(round(DUPLICATE_RATE * n))
    dups = df.iloc[rng.choice(n, size=m, replace=False)]
    df = pd.concat([df, dups]).sort_values("order_id", kind="stable").reset_index(drop=True)
    report["duplicate_rows"] = m
    return df, report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    suppliers = build_suppliers()
    products = build_products(rng, suppliers)
    lead = products.lead_time_days.to_numpy()
    supplier = products.supplier_id.to_numpy()

    rank = rng.permutation(N_SKUS)  # 0 = most popular
    pop = (rank + 1.0) ** -0.65
    pop /= pop.sum()
    cust_region = rng.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)
    cust_p = rng.lognormal(0.0, 1.0, size=N_CUSTOMERS)
    cust_p /= cust_p.sum()

    orders, is_peak = generate_demand(rng, pop, cust_p, cust_region)

    # average daily demand per SKU (the planner's "forecast")
    d = np.zeros(N_SKUS)
    for o in orders:
        if not o["cancelled"]:
            for s, q in o["lines"]:
                d[s] += q
    d /= N_DAYS

    # planted: pick fast-moving SKUs (mostly from late suppliers) to under-stock
    fast = [i for i in range(N_SKUS) if rank[i] < 90]
    late_pool = [i for i in fast if supplier[i] in LATE_SUPPLIERS]
    other_pool = [i for i in fast if supplier[i] not in LATE_SUPPLIERS and lead[i] >= 7]
    n_late = min(N_STOCKOUT_FROM_LATE, len(late_pool))
    chosen = list(rng.choice(late_pool, size=n_late, replace=False))
    chosen += list(rng.choice(other_pool, size=N_STOCKOUT_SKUS - n_late, replace=False))
    is_target = np.zeros(N_SKUS, dtype=bool)
    is_target[chosen] = True

    # Normal SKUs: lead-time demand + a buffer for lumpy trade orders (up to 48 per
    # line) + extra cover for the two late suppliers (the planner has learned to
    # expect delays from them, but the buffer is not enough on the planted SKUs).
    late_buffer = np.where(np.isin(supplier, list(LATE_SUPPLIERS)), LATE_SUPPLIER_BUFFER_DAYS, 0)
    rp_days = RP_SAFETY * lead + RP_FLAT + late_buffer
    rp = (np.ceil(d * rp_days) + LUMP_BUFFER).astype(np.int64)
    init = (np.ceil(d * (rp_days + rng.uniform(3, 20, size=N_SKUS))) + LUMP_BUFFER).astype(np.int64)
    rp[is_target] = np.maximum(np.ceil(d[is_target] * 0.45 * lead[is_target]), 5).astype(np.int64)
    init[is_target] = np.maximum(np.ceil(d[is_target] * (0.8 * lead[is_target] + 3)), 10).astype(np.int64)
    products["reorder_point"] = rp
    products = products[["sku", "description", "category", "unit_cost", "unit_price",
                         "supplier_id", "reorder_point", "lead_time_days"]]

    sim = simulate(rng, orders, products, is_peak, d, rp, init, is_target)
    orders_df, lines_df, inv_df, po_df, labour_df = build_tables(sim, products, suppliers)
    n_stockout = validate(orders_df, lines_df, products, inv_df, suppliers, po_df, labour_df)
    orders_out, defects = inject_defects(rng, orders_df)

    OUT_DIR.mkdir(exist_ok=True)
    tables = {"orders": orders_out, "order_lines": lines_df, "products": products,
              "inventory_daily": inv_df, "suppliers": suppliers,
              "purchase_orders": po_df, "labour_daily": labour_df}
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, date_format="%Y-%m-%d")

    # ---- summary ----
    print(f"Date range: {START.date()} to {END.date()}")
    for name, df in tables.items():
        print(f"  {name + '.csv':22s} {len(df):>7,} rows")
    print(f"Planted defects: {defects}")
    print(f"Stock-out SKUs (on_hand hit 0): {n_stockout}")
    print("Target under-stocked SKUs:", sorted(products.sku[is_target]))
    bl = np.array(sim["backlog"])
    print(f"Open-order backlog: peak {bl.max()} orders on {DATES[bl.argmax()].date()}, "
          f"end of period {bl[-1]}")
    print("Order status:", orders_df.status.value_counts().to_dict())
    rcv = po_df[po_df.received_date.notna()]
    late = (rcv.received_date > rcv.promised_date)
    print("PO late rate by supplier:", late.groupby(rcv.supplier_id).mean().round(2).to_dict())
    lab = labour_df.copy()
    lab["uph"] = lab.units_picked / lab.hours_worked
    lab["peak"] = (lab.date >= PEAK_START) & (lab.date <= PEAK_END)
    print("Units per hour by shift / peak:\n",
          lab.groupby(["shift", "peak"]).uph.mean().round(1).to_string())
    print("Avg weekday utilisation check - units/day: "
          f"{labour_df.groupby('date').units_picked.sum().mean():.0f}")


if __name__ == "__main__":
    main()
