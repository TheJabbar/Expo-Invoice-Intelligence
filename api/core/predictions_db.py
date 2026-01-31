import sqlite3
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from loguru import logger


DB_PATH = Path(os.getenv("INVOICES_DB_PATH", "./invoices.db"))


def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                id TEXT PRIMARY KEY,
                invoice_id TEXT NOT NULL UNIQUE,
                vendor TEXT,
                invoice_no TEXT,
                invoice_date TEXT,
                tax REAL,
                total REAL,
                debit_account TEXT,
                credit_account TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_invoice_id ON invoices(invoice_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON invoices(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vendor ON invoices(vendor)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_debit_account ON invoices(debit_account)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_credit_account ON invoices(credit_account)")

        conn.commit()
        conn.close()

        logger.info(f"Invoices database initialized at {DB_PATH}")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def save_invoice(
    invoice_id: str,
    vendor: str,
    invoice_no: str,
    invoice_date: str,
    tax: float,
    total: float,
    debit_account: str,
    credit_account: str
) -> str:
    try:
        record_id = str(uuid.uuid4())

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO invoices (
                id, invoice_id, vendor, invoice_no, invoice_date,
                tax, total, debit_account, credit_account
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            invoice_id,
            vendor,
            invoice_no,
            invoice_date,
            tax,
            total,
            debit_account,
            credit_account
        ))

        conn.commit()
        conn.close()

        logger.info(f"Invoice saved to database with ID: {record_id}")
        return record_id

    except Exception as e:
        logger.error(f"Error saving invoice to database: {e}")
        raise


def get_invoice(invoice_id: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM invoices WHERE invoice_id = ?", (invoice_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        else:
            return None

    except Exception as e:
        logger.error(f"Error retrieving invoice from database: {e}")
        return None


def get_invoices_by_account(account: str, account_type: str = "debit") -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if account_type == "debit":
            cursor.execute("SELECT * FROM invoices WHERE debit_account = ? ORDER BY timestamp DESC", (account,))
        else:
            cursor.execute("SELECT * FROM invoices WHERE credit_account = ? ORDER BY timestamp DESC", (account,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append(dict(row))

        return results

    except Exception as e:
        logger.error(f"Error retrieving invoices from database: {e}")
        return []


def get_recent_invoices(limit: int = 100) -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM invoices ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append(dict(row))

        return results

    except Exception as e:
        logger.error(f"Error retrieving recent invoices from database: {e}")
        return []


def update_invoice(invoice_id: str, **kwargs):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        values = list(kwargs.values()) + [invoice_id]

        query = f"UPDATE invoices SET {set_clause} WHERE invoice_id = ?"
        cursor.execute(query, values)

        conn.commit()
        conn.close()

        logger.info(f"Updated invoice {invoice_id} in database")

    except Exception as e:
        logger.error(f"Error updating invoice in database: {e}")
        raise


def delete_invoice(invoice_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM invoices WHERE invoice_id = ?", (invoice_id,))
        conn.commit()
        conn.close()

        logger.info(f"Deleted invoice {invoice_id} from database")

    except Exception as e:
        logger.error(f"Error deleting invoice from database: {e}")
        raise


def save_correction(
    prediction_id: str,
    corrected_fields: dict,
    user_id: str
) -> str:
    try:
        # Create a corrections table if it doesn't exist
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create corrections table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                corrected_fields TEXT,
                user_id TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        correction_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO corrections (
                id, prediction_id, corrected_fields, user_id
            ) VALUES (?, ?, ?, ?)
        """, (
            correction_id,
            prediction_id,
            json.dumps(corrected_fields),
            user_id
        ))

        conn.commit()
        conn.close()

        logger.info(f"Correction saved to database with ID: {correction_id}")
        return correction_id

    except Exception as e:
        logger.error(f"Error saving correction to database: {e}")
        raise


# Initialize the database when this module is imported
if not DB_PATH.exists():
    init_db()