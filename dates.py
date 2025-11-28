#!/usr/bin/env python3
"""Debug script to check date ranges in Northwind database."""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/northwind.sqlite")

def check_dates():
    """Check what date ranges exist in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("="*60)
    print("Checking OrderDate ranges in Northwind database")
    print("="*60)
    
    # Get min/max dates
    cursor.execute("SELECT MIN(OrderDate), MAX(OrderDate) FROM Orders")
    min_date, max_date = cursor.fetchone()
    print(f"\nDate Range in Database:")
    print(f"  MIN: {min_date}")
    print(f"  MAX: {max_date}")
    
    # Check 1997 data
    print(f"\n{'='*60}")
    print("Checking 1997 Data")
    print("="*60)
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM Orders 
        WHERE strftime('%Y', OrderDate) = '1997'
    """)
    count_1997 = cursor.fetchone()[0]
    print(f"\nOrders in 1997: {count_1997}")
    
    if count_1997 == 0:
        print("⚠ WARNING: No orders in 1997!")
        cursor.execute("SELECT DISTINCT strftime('%Y', OrderDate) as year FROM Orders ORDER BY year")
        years = [row[0] for row in cursor.fetchall()]
        print(f"Available years: {years}")
    
    # Check Summer 1997 (June)
    print(f"\n{'='*60}")
    print("Checking Summer 1997 (June 1-30)")
    print("="*60)
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM Orders 
        WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30'
    """)
    count_summer = cursor.fetchone()[0]
    print(f"Orders in June 1997: {count_summer}")
    
    # Check Winter 1997 (December)
    print(f"\n{'='*60}")
    print("Checking Winter 1997 (December 1-31)")
    print("="*60)
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM Orders 
        WHERE OrderDate BETWEEN '1997-12-01' AND '1997-12-31'
    """)
    count_winter = cursor.fetchone()[0]
    print(f"Orders in December 1997: {count_winter}")
    
    # Find alternative date ranges with data
    print(f"\n{'='*60}")
    print("Finding Date Ranges with Most Orders")
    print("="*60)
    
    cursor.execute("""
        SELECT strftime('%Y-%m', OrderDate) as month, COUNT(*) as order_count
        FROM Orders
        GROUP BY month
        ORDER BY order_count DESC
        LIMIT 10
    """)
    
    print("\nTop 10 months by order count:")
    for month, count in cursor.fetchall():
        print(f"  {month}: {count} orders")
    
    # Check specific dates around 1997
    print(f"\n{'='*60}")
    print("Checking 2016-2018 Data (your DB shows 2016 dates)")
    print("="*60)
    
    for year in ['2016', '2017', '2018']:
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM Orders 
            WHERE strftime('%Y', OrderDate) = '{year}'
        """)
        count = cursor.fetchone()[0]
        print(f"Orders in {year}: {count}")
    
    conn.close()

if __name__ == '__main__':
    check_dates()