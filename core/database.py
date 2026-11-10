import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Workspaces table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS workspaces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        research_question TEXT NOT NULL,
        hypothesis TEXT,
        research_area TEXT,
        objective TEXT,
        constraints TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # Documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        uploaded_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    )
    """)

    # Experiment plans table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiment_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        plan_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    )
    """)

    # Experiment runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiment_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        plan_id INTEGER,
        dataset_name TEXT NOT NULL,
        experiment_type TEXT NOT NULL,
        config_json TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        interpretation TEXT,
        next_experiment_recommendation TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

# ----------------- Workspaces CRUD ----------------- #

def create_workspace(title: str, research_question: str, hypothesis: str = "",
                     research_area: str = "Data Science", objective: str = "",
                     constraints: str = "") -> int:
    conn = get_connection()
    now = datetime.now().isoformat()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO workspaces (title, research_question, hypothesis, research_area, objective, constraints, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, research_question, hypothesis, research_area, objective, constraints, now, now))
    workspace_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return workspace_id

def get_workspace(workspace_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_workspaces() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM workspaces ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_workspace(workspace_id: int, **kwargs) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    kwargs["updated_at"] = datetime.now().isoformat()
    fields = [f"{k} = ?" for k in kwargs.keys()]
    values = list(kwargs.values()) + [workspace_id]
    cursor.execute(f"UPDATE workspaces SET {', '.join(fields)} WHERE id = ?", values)
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def delete_workspace(workspace_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
    cursor.execute("DELETE FROM documents WHERE workspace_id = ?", (workspace_id,))
    cursor.execute("DELETE FROM experiment_plans WHERE workspace_id = ?", (workspace_id,))
    cursor.execute("DELETE FROM experiment_runs WHERE workspace_id = ?", (workspace_id,))
    conn.commit()
    conn.close()
    return True

# ----------------- Documents CRUD ----------------- #

def add_document(workspace_id: int, filename: str, file_type: str, file_path: str, chunk_count: int = 0) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO documents (workspace_id, filename, file_type, file_path, chunk_count, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (workspace_id, filename, file_type, file_path, chunk_count, now))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def list_documents(workspace_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE workspace_id = ? ORDER BY uploaded_at DESC", (workspace_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_document(doc_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return True

# ----------------- Experiment Plans CRUD ----------------- #

def save_experiment_plan(workspace_id: int, plan_dict: Dict[str, Any]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO experiment_plans (workspace_id, plan_json, created_at)
        VALUES (?, ?, ?)
    """, (workspace_id, json.dumps(plan_dict), now))
    plan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return plan_id

def get_latest_plan(workspace_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM experiment_plans WHERE workspace_id = ? ORDER BY created_at DESC LIMIT 1
    """, (workspace_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        res = dict(row)
        res["plan"] = json.loads(res["plan_json"])
        return res
    return None

def list_plans(workspace_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiment_plans WHERE workspace_id = ? ORDER BY created_at DESC", (workspace_id,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["plan"] = json.loads(d["plan_json"])
        result.append(d)
    return result

# ----------------- Experiment Runs CRUD ----------------- #

def save_experiment_run(workspace_id: int, dataset_name: str, experiment_type: str,
                        config_dict: Dict[str, Any], metrics_dict: Dict[str, Any],
                        plan_id: Optional[int] = None, interpretation: str = "",
                        next_recommendation: str = "") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO experiment_runs (
            workspace_id, plan_id, dataset_name, experiment_type,
            config_json, metrics_json, interpretation, next_experiment_recommendation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        workspace_id, plan_id, dataset_name, experiment_type,
        json.dumps(config_dict), json.dumps(metrics_dict),
        interpretation, next_recommendation, now
    ))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def update_run_interpretation(run_id: int, interpretation: str, next_recommendation: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE experiment_runs
        SET interpretation = ?, next_experiment_recommendation = ?
        WHERE id = ?
    """, (interpretation, next_recommendation, run_id))
    conn.commit()
    conn.close()

def get_experiment_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiment_runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        res = dict(row)
        res["config"] = json.loads(res["config_json"])
        res["metrics"] = json.loads(res["metrics_json"])
        return res
    return None

def list_experiment_runs(workspace_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if workspace_id:
        cursor.execute("SELECT * FROM experiment_runs WHERE workspace_id = ? ORDER BY created_at DESC", (workspace_id,))
    else:
        cursor.execute("SELECT * FROM experiment_runs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d["config_json"])
        d["metrics"] = json.loads(d["metrics_json"])
        results.append(d)
    return results

def delete_experiment_run(run_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiment_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
    return True
