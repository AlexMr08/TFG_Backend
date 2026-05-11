import os
import jwt
from openai import OpenAI
from datetime import datetime, timedelta, timezone

# --- RUTAS ---
# Asegúrate de que esta carpeta apunta a donde descomprimiste el dataset
CARPETA_IMAGENES = "D:\wikiart" 
CSV_PATH = "D:/wikiart/wclasses.csv"
DB_PATH = "./arte_db"


SECRET_KEY = "lleva_la_tarara_un_vestido_blanco"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 86400  # 1 day

# --- CLIENTE VLLM ---

client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="your_api_key_here"
)

# --- DICCIONARIO DE MAPEO (IDs a Nombres) ---
# Copia aquí el diccionario que hicimos antes
ID_TO_LABEL = {
0: 'Unknown Artist', 1: 'Boris Kustodiev', 2: 'Camille Pissarro', 3: 'Childe Hassam', 4: 'Claude Monet', 
  5: 'Edgar Degas', 6: 'Eugene Boudin', 7: 'Gustave Dore', 8: 'Ilya Repin', 9: 'Ivan Aivazovsky', 
  10: 'Ivan Shishkin', 11: 'John Singer Sargent', 12: 'Marc Chagall', 13: 'Martiros Saryan', 14: 'Nicholas Roerich', 
  15: 'Pablo Picasso', 16: 'Paul Cezanne', 17: 'Pierre Auguste Renoir', 18: 'Pyotr Konchalovsky', 19: 'Raphael Kirchner', 
  20: 'Rembrandt', 21: 'Salvador Dali', 22: 'Vincent van Gogh', 23: 'Hieronymus Bosch', 24: 'Leonardo da Vinci', 
  25: 'Albrecht Durer', 26: 'Edouard Cortes', 27: 'Sam Francis', 28: 'Juan Gris', 29: 'Lucas Cranach the Elder', 
  30: 'Paul Gauguin', 31: 'Konstantin Makovsky', 32: 'Egon Schiele', 33: 'Thomas Eakins', 34: 'Gustave Moreau', 
  35: 'Francisco Goya', 36: 'Edvard Munch', 37: 'Henri Matisse', 38: 'Fra Angelico', 39: 'Maxime Maufra', 
  40: 'Jan Matejko', 41: 'Mstislav Dobuzhinsky', 42: 'Alfred Sisley', 43: 'Mary Cassatt', 44: 'Gustave Loiseau', 
  45: 'Fernando Botero', 46: 'Zinaida Serebriakova', 47: 'Georges Seurat', 48: 'Isaac Levitan', 49: 'Joaquin Sorolla', 
  50: 'Jacek Malczewski', 51: 'Berthe Morisot', 52: 'Andy Warhol', 53: 'Arkhip Kuindzhi', 54: 'Niko Pirosmani', 
  55: 'James Tissot', 56: 'Vasily Polenov', 57: 'Valentin Serov', 58: 'Pietro Perugino', 59: 'Pierre Bonnard', 
  60: 'Ferdinand Hodler', 61: 'Bartolome Esteban Murillo', 62: 'Giovanni Boldini', 63: 'Henri Martin', 64: 'Gustav Klimt', 
  65: 'Vasily Perov', 66: 'Odilon Redon', 67: 'Tintoretto', 68: 'Gene Davis', 69: 'Raphael', 
  70: 'John Henry Twachtman', 71: 'Henri de Toulouse Lautrec', 72: 'Antoine Blanchard', 73: 'David Burliuk', 74: 'Camille Corot', 
  75: 'Konstantin Korovin', 76: 'Ivan Bilibin', 77: 'Titian', 78: 'Maurice Prendergast', 79: 'Edouard Manet', 
  80: 'Peter Paul Rubens', 81: 'Aubrey Beardsley', 82: 'Paolo Veronese', 83: 'Joshua Reynolds', 84: 'Kuzma Petrov Vodkin', 
  85: 'Gustave Caillebotte', 86: 'Lucian Freud', 87: 'Michelangelo', 88: 'Dante Gabriel Rossetti', 89: 'Felix Vallotton', 
  90: 'Nikolay Bogdanov Belsky', 91: 'Georges Braque', 92: 'Vasily Surikov', 93: 'Fernand Leger', 94: 'Konstantin Somov', 
  95: 'Katsushika Hokusai', 96: 'Sir Lawrence Alma Tadema', 97: 'Vasily Vereshchagin', 98: 'Ernst Ludwig Kirchner', 99: 'Mikhail Vrubel', 
  100: 'Orest Kiprensky', 101: 'William Merritt Chase', 102: 'Aleksey Savrasov', 103: 'Hans Memling', 104: 'Amedeo Modigliani', 
  105: 'Ivan Kramskoy', 106: 'Utagawa Kuniyoshi', 107: 'Gustave Courbet', 108: 'William Turner', 109: 'Theo van Rysselberghe', 
  110: 'Joseph Wright', 111: 'Edward Burne Jones', 112: 'Koloman Moser', 113: 'Viktor Vasnetsov', 114: 'Anthony van Dyck', 
  115: 'Raoul Dufy', 116: 'Frans Hals', 117: 'Hans Holbein the Younger', 118: 'Ilya Mashkov', 119: 'Henri Fantin Latour', 
  120: 'M.C. Escher', 121: 'El Greco', 122: 'Mikalojus Ciurlionis', 123: 'James McNeill Whistler', 124: 'Karl Bryullov', 
  125: 'Jacob Jordaens', 126: 'Thomas Gainsborough', 127: 'Eugene Delacroix', 128: 'Canaletto',
  # Géneros
  129: 'Abstract Painting', 130: 'Cityscape', 131: 'Genre Painting', 132: 'Illustration', 133: 'Landscape', 
  134: 'Nude Painting', 135: 'Portrait', 136: 'Religious Painting', 137: 'Sketch and Study', 138: 'Still Life', 
  139: 'Unknown Genre',
  # Estilos
  140: 'Abstract Expressionism', 141: 'Action Painting', 142: 'Analytical Cubism', 143: 'Art Nouveau', 144: 'Baroque', 
  145: 'Color Field Painting', 146: 'Contemporary Realism', 147: 'Cubism', 148: 'Early Renaissance', 149: 'Expressionism', 
  150: 'Fauvism', 151: 'High Renaissance', 152: 'Impressionism', 153: 'Mannerism Late Renaissance', 154: 'Minimalism', 
  155: 'Naive Art Primitivism', 156: 'New Realism', 157: 'Northern Renaissance', 158: 'Pointillism', 159: 'Pop Art', 
  160: 'Post Impressionism', 161: 'Realism', 162: 'Rococo', 163: 'Romanticism', 164: 'Symbolism', 
  165: 'Synthetic Cubism', 166: 'Ukiyo e', 167: 'Unknown Style'
}

def create_access_token(user_id: str):
    now = datetime.now(timezone.utc)
    expire = now + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)

    payload = {
        "sub": user_id,       # ID interno (UUID)
        "exp": expire,
        "iat": now,          # Issued At
    }
    # Debug: show the exact expiration used to sign this token
    print("DEBUG create_access_token exp:", int(expire.timestamp()), " iat: ", int(now.timestamp()))
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)