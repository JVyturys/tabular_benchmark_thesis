import wrds

def main():
    print("[*] Verbinde mit WRDS...")
    db = wrds.Connection()
    
    print("\n[*] Suche nach ESG-bezogenen Bibliotheken...")
    all_libs = db.list_libraries()
    
    keywords = ['esg', 'tr_', 'asset4', 'refinitiv', 'thomson']
    potential_libs = [lib for lib in all_libs if any(k in lib.lower() for k in keywords)]
    
    if not potential_libs:
        print("[-] Keine ESG-Bibliotheken gefunden.")
        return
        
    print(f"[+] Untersuche Bibliotheken: {potential_libs}")
    
    for lib in potential_libs:
        print(f"\n--- Tabellen in Bibliothek '{lib}' ---")
        try:
            tables = db.list_tables(lib)
            for table in tables:
                print(f" - {table}")
        except Exception as e:
            print(f" Konnte nicht auf '{lib}' zugreifen: {e}")

if __name__ == "__main__":
    main()