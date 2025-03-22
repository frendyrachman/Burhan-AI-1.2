import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fungsi untuk mendapatkan data hadis dari sebuah URL buku
def get_hadiths_from_book(url):
    print(f"Mengakses URL buku: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    hadiths_data = []
    
    # Ambil judul buku
    book_title_element = soup.find('div', class_='book_page_english_name')
    book_title = book_title_element.text.strip() if book_title_element else "Judul Tidak Ditemukan"
    print(f"Judul buku ditemukan: {book_title}")
    
    # Loop melalui setiap hadis
    hadith_containers = soup.find_all('div', class_='actualHadithContainer')
    print(f"Menemukan {len(hadith_containers)} hadis dalam buku ini.")
    
    for hadith_container in hadith_containers:
        # Ambil narator hadis
        narrator = hadith_container.find('div', class_='hadith_narrated')
        narrator_text = narrator.text.strip() if narrator else "N/A"
        
        # Ambil teks hadis
        hadith_text = hadith_container.find('div', class_='text_details')
        hadith_text = hadith_text.text.strip() if hadith_text else "N/A"
        
        # Ambil referensi hadis
        reference = hadith_container.find('div', class_='hadith_reference_sticky')
        reference_text = reference.text.strip() if reference else "N/A"
        
        # Simpan data ke dalam list
        hadiths_data.append({
            'Rawi': url.split('/')[-2].capitalize(),  # Nama periwayat dari URL
            'Chapter': book_title,  # Judul buku
            'Reference': reference_text,  # Referensi hadis
            'Narator': narrator_text,  # Narator hadis
            'Hadiths Text': hadith_text  # Teks hadis
        })
    
    print(f"Berhasil mengumpulkan {len(hadiths_data)} hadis dari buku ini.\n")
    return hadiths_data

# Fungsi untuk mendapatkan semua buku dari sebuah periwayat
def get_books_from_author(base_url):
    books_data = []
    book_number = 1
    
    print(f"\nMemulai scraping untuk periwayat: {base_url}")
    
    while True:
        url = f"{base_url}/{book_number}"
        print(f"Mengecek URL buku: {url}")
        response = requests.get(url)
        
        # Jika halaman tidak ditemukan, hentikan loop
        if response.status_code != 200:
            print(f"Buku dengan nomor {book_number} tidak ditemukan. Menghentikan proses untuk periwayat ini.\n")
            break
        
        # Ambil data hadis dari buku tersebut
        books_data.extend(get_hadiths_from_book(url))
        book_number += 1
    
    print(f"Total {len(books_data)} hadis ditemukan untuk periwayat {base_url}.\n")
    return books_data

# List URL periwayat
authors_urls = [
    "https://sunnah.com/bukhari",
    "https://sunnah.com/muslim",
    "https://sunnah.com/nasai",
    "https://sunnah.com/abudawud",
    "https://sunnah.com/ibnmajah",
    "https://sunnah.com/malik",
    "https://sunnah.com/ahmad",
    "https://sunnah.com/riyadussalihin",
    "https://sunnah.com/adab",
    "https://sunnah.com/shamail",
    "https://sunnah.com/mishkat",
    "https://sunnah.com/hisn"
]

# Kumpulkan semua data hadis
all_hadiths = []
for author_url in authors_urls:
    print(f"\n=== Memulai scraping untuk periwayat: {author_url} ===")
    all_hadiths.extend(get_books_from_author(author_url))

# Buat dataframe dari data yang dikumpulkan
df = pd.DataFrame(all_hadiths, columns=['Rawi', 'Chapter', 'Reference', 'Narator', 'Hadiths Text'])

# Simpan dataframe ke dalam file CSV
df.to_csv('hadiths_data.csv', index=False, encoding='utf-8')
print("\n=== Proses scraping selesai ===")
print(f"Total {len(df)} hadis berhasil dikumpulkan dan disimpan ke dalam file hadiths_data.csv.")