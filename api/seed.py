"""
Seed initial users — BRD §12, D-018.

ensure_users_seeded(repo) is idempotent: if the users table already has rows
it exits immediately. Call once from lifespan() on API startup.

One user per BRD RBAC role (§12):
  commercial_head, planner, sales_manager, sop_chair

All writes go through the repository (Rule 1 — no direct sqlite3).

Note: uses bcrypt directly (not passlib) because passlib 1.7.4 is incompatible
with bcrypt >= 4.0.0.
"""
import bcrypt


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


_SEED_USERS = [
    {
        "user_id":              "commercial_head_01",
        "display_name":         "Commercial Head (Demo)",
        "role":                 "commercial_head",
        "assigned_states_json": "[]",
        "password_hash":        _hash("ch-demo-2024"),
    },
    {
        "user_id":              "planner_01",
        "display_name":         "Planner 1 (Demo)",
        "role":                 "planner",
        "assigned_states_json": "[]",
        "password_hash":        _hash("pl-demo-2024"),
    },
    {
        "user_id":              "planner_02",
        "display_name":         "Planner 2 (Demo)",
        "role":                 "planner",
        "assigned_states_json": "[]",
        "password_hash":        _hash("pl2-demo-2024"),
    },
    {
        "user_id":              "sales_mgr_01",
        "display_name":         "Sales Manager (Demo)",
        "role":                 "sales_manager",
        "assigned_states_json": "[]",
        "password_hash":        _hash("sm-demo-2024"),
    },
    {
        "user_id":              "sop_chair_01",
        "display_name":         "S&OP Chair (Demo)",
        "role":                 "sop_chair",
        "assigned_states_json": "[]",
        "password_hash":        _hash("sop-demo-2024"),
    },
]


def ensure_users_seeded(repo) -> None:
    """Seed 5 demo users if the users table is empty. Idempotent."""
    existing = repo.query("users")
    if existing:
        print(f"[seed] users table already has {len(existing)} rows — skipping seed.")
        return

    repo.upsert("users", _SEED_USERS)
    print(f"[seed] Seeded {len(_SEED_USERS)} demo users.")
