# Power BI Build Guide — Vahaan Bazaar Analytics

This turns the CSVs in `powerbi/data/` into the same dashboard shown in the
HTML preview (`Dashboard_Preview.html`) — about 15-20 minutes end to end.
All monetary fields are Indian Rupees (INR).

## 1. Import the data

Power BI Desktop → **Get Data → Text/CSV**, import all 8 files from
`powerbi/data/`:

| File | Role |
|---|---|
| `dim_date.csv` | Date dimension |
| `dim_vehicle.csv` | Vehicle dimension (one row per vehicle in inventory) |
| `dim_advocate.csv` | Sales advocate dimension |
| `dim_center.csv` | Recon (reconditioning) center dimension |
| `fact_orders.csv` | Sales fact (one row per sold vehicle) |
| `fact_inspections.csv` | QC fact (one row per vehicle inspection) |
| `fact_order_economics.csv` | Gross-profit fact (one row per sold vehicle) |
| `fact_returns.csv` | Returns fact (one row per 7-day money-back return) |

In Power Query Editor, confirm date columns imported as **Date** type
(`dim_date[date_key]`, `fact_orders[order_date]`, `fact_inspections[inspection_date]`,
`fact_order_economics[order_date]`, `dim_vehicle[acquisition_date]`,
`fact_returns[return_date]`) — CSV import sometimes leaves these as text.
Fix via the column-type icon if needed, then set `sale_price_inr`,
`acquisition_cost_inr`, `gross_profit_inr`, `refund_amount_inr`, etc. to
**Fixed decimal number** or **Whole number**, then **Close & Apply**.

Set the currency format on every INR column/measure to a custom format
`₹ #,##0` (Power BI's Indian locale format under Modeling → Format →
Currency → ₹ Indian Rupee also works and applies lakh/crore grouping).

## 2. Build the relationships (Model view)

| From | To | Cardinality |
|---|---|---|
| `dim_date[date_key]` | `fact_orders[order_date]` | 1 → many |
| `dim_date[date_key]` | `fact_inspections[inspection_date]` | 1 → many |
| `dim_vehicle[vehicle_id]` | `fact_orders[vehicle_id]` | 1 → many |
| `dim_vehicle[vehicle_id]` | `fact_inspections[vehicle_id]` | 1 → many |
| `dim_advocate[advocate_id]` | `fact_orders[advocate_id]` | 1 → many |
| `dim_center[center_id]` | `dim_vehicle[center_id]` | 1 → many |
| `fact_orders[order_id]` | `fact_order_economics[order_id]` | 1 → 1 |
| `fact_orders[order_id]` | `fact_returns[order_id]` | 1 → many |

`dim_vehicle[vehicle_id]` → `fact_orders[vehicle_id]` is 1-to-many even
though each vehicle sells at most once (0 or 1 orders) — this is what
makes the **Unsold Inventory Count** measure work: unsold vehicles exist
in `dim_vehicle` with no matching row in `fact_orders`.

All relationships are single-direction filter (dimension → fact), the
standard star-schema pattern — don't switch any to bidirectional unless a
specific visual needs it.

Right-click `dim_date` → **Mark as Date Table**, using `date_key` as the
date column. This enables the time-intelligence DAX in `DAX_Measures.md`
(`DATEADD`, running totals, etc.).

## 3. Add the measures

Paste every measure from `DAX_Measures.md` (Modeling → New Measure). Doing
this before building visuals means every visual below can just drag in a
measure by name instead of raw columns.

## 4. Report pages

### Page 1 — Executive Overview

- **KPI cards** across the top: `[Total Revenue]`, `[Units Sold]`,
  `[Avg Gross Profit per Unit (GPU)]`, `[Inspection Pass Rate]`.
- **Line chart**: `[Cumulative YTD Revenue]` (Y) by `dim_date[year_month]`
  (X) — the running-total view, mirrors SQL Q12.
- **Bar chart**: `[Total Revenue]` by `dim_vehicle[body_type]`, sorted
  descending — mirrors SQL Q1.
- **Bar chart**: `[Units Sold]` split by `dim_vehicle[body_type]` x
  `dim_date[season]` (Festive Season / Monsoon / Rest of Year) — mirrors
  SQL Q9's seasonal comparison.
- **Slicers** (top of page, one row): `dim_date[year]`,
  `dim_advocate[region]`, `dim_vehicle[body_type]`.

### Page 2 — Advocate & Commission Performance

- **Table or matrix**: rows = `dim_advocate[advocate_name]`; columns =
  `[Units Sold]`, `monthly_unit_quota`, `[% of Monthly Quota]`,
  `[Quota Status]`, `[Commission Payout]`. Add a **data bar** conditional
  format on `[% of Monthly Quota]` (right-click the measure →
  Conditional formatting → Data bars) so quota attainment reads at a
  glance, matching the Excel Commission_Calculator's color scale.
- **Bar chart**: `[Total Revenue]` by `dim_advocate[region]`.
- **Slicer**: a month selector on `dim_date[year_month]` — this is the
  direct analog of the Excel Commission_Calculator's month dropdown; pick
  the same month in both tools and the advocate totals should match
  exactly.
- Add a card showing a count of advocates where `[Quota Status]` =
  "Behind Quota" to headline attention-needed advocates (mirrors SQL Q3).
- Optional: a card or table row for Ishita Bansal (North region) will show
  zero recent activity — she's the seeded early-churn case that SQL Q10's
  anti-join is built to surface.

### Page 3 — Reconditioning & Quality Control

- **KPI cards**: `[Inspections]`, `[Failed Inspections]`,
  `[Inspection Pass Rate]`, `[Avg Mechanical Score]`.
- **Bar chart**: `[Inspection Pass Rate]` by `dim_center[center_name]` —
  facility comparison (Bengaluru vs. Gurugram recon hubs), mirrors SQL Q5.
- **Bar chart**: `[Inspection Pass Rate]` by `fact_inspections[technician_name]`.
- **Table**: failed inspections only (filter `overall_result = "Fail"`)
  with vehicle, center, technician, inspection date, mechanical/cosmetic
  scores, and road test result — for a QA reviewer to work from directly,
  mirrors SQL Q6's root-cause diagnostics.

### Page 4 — Profitability & Returns

- **KPI cards**: `[Total Gross Profit]`, `[Avg Gross Profit per Unit (GPU)]`,
  `[% Units Sold at a Loss]`, `[Return Rate %]`, `[Total Refunded]`.
- **Bar chart**: `[Avg Gross Profit per Unit (GPU)]` by
  `dim_vehicle[body_type]` — mirrors SQL Q7.
- **Bar chart**: `[Avg Days Acquisition to Sale]` by `dim_vehicle[body_type]`
  — the inventory-turn-speed view, mirrors SQL Q8.
- **Bar chart**: `[Return Rate %]` by `dim_vehicle[make]` — the "rare and
  interesting" angle unique to this domain (India's 7-day money-back
  return policy on used-vehicle purchases), mirrors SQL Q13.
- **Table**: `fact_returns` detail (order, make/model, return reason,
  refund amount) for the ops team to review.

## 5. Formatting & final polish

- Apply the categorical palette used in `Dashboard_Preview.html`
  consistently: blue `#2a78d6` as the primary series, orange `#eb6834` as
  the secondary series, aqua `#1baf7a` for a third series where needed;
  status colors `#0ca30c` (good/On-Above Quota), `#fab219`
  (warning/Near Quota), `#d03b3b` (critical/Behind Quota or Fail).
- Set the report theme's background to the same off-white chart surface
  (`#fcfcfb`) used in the HTML preview for visual consistency across tools.
- Every INR figure should use the `₹ #,##0` custom format (or Power BI's
  built-in Indian Rupee currency format) — never plain `$` or unformatted
  numbers.

## 6. Verifying the build

Pick any month in the Page 2 slicer and compare the advocate table to the
same month in the Excel `Commission_Calculator` sheet (set the same month
in its dropdown) — the Units Sold, Revenue, and Commission Payout columns
should match exactly, because both tools read from the same underlying
SQL views (`vw_orders_detail`, `vw_advocate_monthly`). This is the direct,
checkable proof that all three tools are genuinely interconnected rather
than just visually similar.

