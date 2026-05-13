from typing import TypedDict


class Skill(TypedDict):
    """A skill that can be progressively disclosed to the agent."""
    name: str  # Unique identifier for the skill
    description: str  # 1-2 sentence description to show in system prompt
    content: str  # Full skill content with detailed instructions


SKILLS: list[Skill] = [
    {
        "name": "purchase_analytics",
        "description": (
            "Analyze purchase orders, vendor spend, procurement trends, product "
            "buying patterns, and supplier performance."
        ),
        "content": """
Use this skill for questions about purchases, procurement, vendors, purchase orders,
RFQs, received quantities, supplier bills, purchase costs, and buying trends.

Typical tables to inspect first:
- purchase_order
- purchase_order_line
- res_partner
- product_product
- product_template
- account_move
- account_move_line

Guidance:
- Use purchase_order.amount_total for total purchase value when available.
- Filter confirmed purchases with states such as 'purchase' or 'done'.
- Use purchase_order.date_order for purchase timing.
- Use purchase_order_line.product_qty for ordered quantities.
- Join vendors through purchase_order.partner_id -> res_partner.id.
- Join products through purchase_order_line.product_id -> product_product.id,
  then product_product.product_tmpl_id -> product_template.id.
- If vendor bills are needed, inspect account_move records with move_type such
  as 'in_invoice'.
- Keep all queries read-only and select only columns needed for the answer.
""".strip(),
    },
    {
        "name": "budget_analytics",
        "description": (
            "Analyze budgets, planned versus actual spend, budget utilization, "
            "variance, departments, accounts, and time-period comparisons."
        ),
        "content": """
Use this skill for questions about budgets, spending limits, planned amounts,
actual expenses, utilization, over-budget areas, under-budget areas, and variance.

Typical tables to inspect first:
- crossovered_budget
- crossovered_budget_lines
- account_budget_post
- account_account
- account_move
- account_move_line
- analytic_account
- account_analytic_account

Guidance:
- Compare planned budget amounts against actual posted accounting entries.
- Use date_from and date_to fields when available for budget periods.
- Group by budget, account, analytic account, department, or month depending on
  the user's question.
- Treat draft accounting records carefully; prefer posted records when the user
  asks for actual spend.
- Clearly label variance as actual minus planned, or planned minus actual,
  depending on the query wording.
- Keep all queries read-only and select only columns needed for the answer.
""".strip(),
    },
    {
        "name": "sales_analytics",
        "description": (
            "Analyze sales orders, revenue, customers, products, sales teams, "
            "quotations, invoices, and sales performance trends."
        ),
        "content": """
Use this skill for questions about sales, revenue, customers, quotations, sales
orders, invoices, product sales, salespeople, sales teams, and sales trends.

Typical tables to inspect first:
- sale_order
- sale_order_line
- res_partner
- product_product
- product_template
- account_move
- account_move_line
- crm_team
- res_users

Guidance:
- Use sale_order.amount_total for sales order totals when available.
- Filter confirmed sales with states such as 'sale' or 'done'.
- Use sale_order.date_order for sales order timing.
- Use sale_order_line.product_uom_qty for sold quantities.
- Join customers through sale_order.partner_id -> res_partner.id.
- Join products through sale_order_line.product_id -> product_product.id,
  then product_product.product_tmpl_id -> product_template.id.
- For invoiced revenue, inspect account_move records with move_type such as
  'out_invoice' and posted state.
- Keep all queries read-only and select only columns needed for the answer.
""".strip(),
    },
    {
        "name": "human_resource_manager",
        "description": (
            "Analyze employees, departments, jobs, attendance, leaves, payroll, "
            "recruitment, and HR workforce metrics."
        ),
        "content": """
Use this skill for questions about employees, departments, job positions,
attendance, leaves, payroll, recruitment, contracts, workforce counts, and HR
performance indicators.

Typical tables to inspect first:
- hr_employee
- hr_department
- hr_job
- hr_attendance
- hr_leave
- hr_leave_type
- hr_contract
- hr_payslip
- hr_applicant

Guidance:
- Use hr_employee for employee master data and active workforce counts.
- Join departments through hr_employee.department_id -> hr_department.id.
- Join job positions through hr_employee.job_id -> hr_job.id.
- Use hr_attendance for check-in/check-out and worked-hour analysis.
- Use hr_leave and hr_leave_type for absence, vacation, and leave analytics.
- Use payroll tables only when the user asks about salaries, payslips, or payroll.
- Avoid exposing sensitive personal details unless directly required by the
  user's question.
- Keep all queries read-only and select only columns needed for the answer.
""".strip(),
    },
    {
        "name": "inventory_management",
        "description": (
            "Analyze stock levels, product movements, warehouses, locations, "
            "inventory valuation, receipts, deliveries, and reorder needs."
        ),
        "content": """
Use this skill for questions about stock, inventory, warehouses, locations,
receipts, deliveries, transfers, on-hand quantities, product movement, and
inventory valuation.

Typical tables to inspect first:
- stock_quant
- stock_move
- stock_move_line
- stock_picking
- stock_picking_type
- stock_location
- stock_warehouse
- product_product
- product_template

Guidance:
- Use stock_quant for current on-hand quantities by product and location.
- Use stock_move and stock_move_line for historical stock movements.
- Use stock_picking for receipts, deliveries, and transfers.
- Join products through product_product.id and product_template for readable names.
- Join locations through stock_location for warehouse and internal location context.
- Distinguish on-hand, reserved, incoming, and outgoing quantities when relevant.
- Keep all queries read-only and select only columns needed for the answer.
""".strip(),
    },
]
