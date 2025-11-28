"""SQLite database access and schema introspection."""
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import re

DB_PATH = Path("data/northwind.sqlite")

def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def get_schema() -> str:
    """Get database schema information."""
    conn = get_connection()
    cursor = conn.cursor()
    
    schema_parts = []
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    for (table_name,) in tables:
        # Skip internal tables
        if table_name.startswith('sqlite_'):
            continue
        
        schema_parts.append(f"\nTable: {table_name}")
        
        # Get columns
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        
        col_info = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            col_info.append(f"  {col_name} ({col_type})")
        
        schema_parts.append("\n".join(col_info))
    
    conn.close()
    
    return "\n".join(schema_parts)

def execute_query(sql: str) -> Dict[str, Any]:
    """Execute SQL query and return results."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(sql)
        
        # Get column names
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        # Get rows
        rows = cursor.fetchall()
        
        # Extract tables used (for citations)
        tables_used = extract_tables_from_sql(sql)
        
        conn.close()
        
        return {
            'columns': columns,
            'rows': rows,
            'error': None,
            'tables_used': tables_used
        }
    
    except Exception as e:
        return {
            'columns': [],
            'rows': [],
            'error': str(e),
            'tables_used': []
        }

def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    # Normalize SQL
    sql_upper = sql.upper()
    
    tables = []
    
    # Common table names in Northwind
    possible_tables = [
        'Orders', 'Order Details', 'Products', 'Customers', 
        'Categories', 'Suppliers', 'Employees', 'Shippers',
        'orders', 'order_details', 'products', 'customers'
    ]
    
    for table in possible_tables:
        # Check if table appears in SQL
        if table.upper() in sql_upper or f'"{table}"' in sql:
            # Use canonical name
            if table.lower() in ['orders']:
                tables.append('Orders')
            elif table.lower() in ['order details', 'order_details']:
                tables.append('Order Details')
            elif table.lower() in ['products']:
                tables.append('Products')
            elif table.lower() in ['customers']:
                tables.append('Customers')
            elif table.lower() in ['categories']:
                tables.append('Categories')
            elif table.lower() in ['suppliers']:
                tables.append('Suppliers')
    
    return list(set(tables))

def test_connection():
    """Test database connection."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Orders")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"✓ Database connected. Found {count} orders.")
        return True
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False