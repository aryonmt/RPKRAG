from document_qa.storage.sqlite import get_db_connection

def print_latest_conversations():
    conn = get_db_connection()
    # ۵ گفتگوی آخر رو به ترتیب زمان (جدیدترین اول) می‌گیریم
    rows = conn.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC LIMIT 5").fetchall()
    conn.close()

    if not rows:
        print("❌ هیچ گفتگویی در دیتابیس وجود ندارد.")
        return

    print("✅ ۵ گفتگوی آخر شما:")
    print("-" * 50)
    for r in rows:
        print(f"ID: {r['id']}")
        print(f"عنوان: {r['title']}")
        print(f"زمان ساخت: {r['created_at']}")
        print("-" * 50)

if __name__ == "__main__":
    print_latest_conversations()