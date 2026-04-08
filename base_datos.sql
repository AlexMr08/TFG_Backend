Version 0.4

ALTER TABLE images
ADD CONSTRAINT uq_images_local_route UNIQUE (local_route);

ALTER TABLE images
ADD COLUMN IF NOT EXISTS artist_id UUID,
ADD CONSTRAINT fk_images_artist
FOREIGN KEY (artist_id) REFERENCES artists(id);

ALTER TABLE images
ADD COLUMN IF NOT EXISTS genre_id UUID,
ADD CONSTRAINT fk_images_genre
FOREIGN KEY (genre_id) REFERENCES genres(id);

ALTER TABLE images
ADD COLUMN IF NOT EXISTS style_id UUID,
ADD CONSTRAINT fk_images_style
FOREIGN KEY (style_id) REFERENCES styles(id);

CREATE TABLE IF NOT EXISTS artists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_artists_name UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS genres (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_genres_name UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS styles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_styles_name UNIQUE (name)
);

ALTER TABLE messages
ADD COLUMN question_id UUID,
ADD CONSTRAINT fk_messages_qid FOREIGN KEY (question_id) REFERENCES messages(id) ON DELETE CASCADE;

ALTER TABLE messages
ADD COLUMN status TEXT DEFAULT 'CREATED';

Version: 0.3
Añadidada a images una foreign key al usuario que la subió o en caso de ser del sistema, null

CREATE TABLE IF NOT EXISTS users(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    firebase_uid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    surname TEXT,
	email TEXT UNIQUE NOT NULL,
    profile_icon TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE IF NOT EXISTS images(
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	local_route TEXT NOT NULL,
	name TEXT DEFAULT 'Unknown',
	artist TEXT DEFAULT 'Unknown',
	style TEXT DEFAULT 'Unknown',
	genre TEXT DEFAULT 'Unknown',
	year TEXT DEFAULT 'Unknown',
	owner_id UUID,
	CONSTRAINT fk_owner FOREIGN KEY (owner_id) REFERENCES users(id)
)

CREATE TABLE IF NOT EXISTS chats(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    topic TEXT,
    image_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id),
	CONSTRAINT fk_image FOREIGN KEY (image_id) REFERENCES images(id)
)

ALTER TABLE chats 
ADD CONSTRAINT unique_user_image 
UNIQUE (user_id, image_id);


CREATE TABLE IF NOT EXISTS messages(
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	chat_id UUID NOT NULL,
	response BOOL DEFAULT false,
	content TEXT,
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT fk_messages_chat FOREIGN KEY (chat_id) REFERENCES chats(id)
)

CREATE TABLE IF NOT EXISTS related_images(
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	image_id UUID NOT NULL,
	message_id UUID NOT NULL,
	similarity NUMERIC(5, 2) NOT NULL,
	CONSTRAINT fk_relatedImages_message FOREIGN KEY (message_id) REFERENCES messages(id),
	CONSTRAINT fk_relatedImages_image FOREIGN KEY (image_id) REFERENCES images(id)
)


Version: 0.2
Añadida la tabla de chats, con sus respectivas claves foráneas a users e images. Además, se ha cambiado el tipo de id a UUID para ambas tablas, y se ha añadido la función gen_random_uuid() para generar automáticamente los UUIDs.
ACTUALMENTE LAS IMAGENES SIGUEN USANDO INT EN LA BD, POR LO QUE SE DEBE CAMBIAR A UUID PARA MANTENER LA CONSISTENCIA.

CREATE TABLE IF NOT EXISTS users(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    firebase_uid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    surname TEXT,
	email TEXT UNIQUE NOT NULL,
    profile_icon TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE IF NOT EXISTS images(
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	local_route TEXT NOT NULL,
	name TEXT DEFAULT 'Unknown',
	artist TEXT DEFAULT 'Unknown',
	style TEXT DEFAULT 'Unknown',
	genre TEXT DEFAULT 'Unknown',
	year TEXT DEFAULT 'Unknown'
)

CREATE TABLE IF NOT EXISTS chats(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    topic TEXT,
    image_id INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id),
	CONSTRAINT fk_image FOREIGN KEY (image_id) REFERENCES images(id)
)


Version: 0.1

CREATE TABLE IF NOT EXISTS users(
    id SERIAL PRIMARY KEY,
    firebase_uid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    surname TEXT,
	email TEXT UNIQUE NOT NULL,
    profile_icon TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE IF NOT EXISTS images(
	id SERIAL PRIMARY KEY,
	local_route TEXT NOT NULL UNIQUE,
	name TEXT DEFAULT 'Unknown',
	artist TEXT DEFAULT 'Unknown',
	style TEXT DEFAULT 'Unknown',
	genre TEXT DEFAULT 'Unknown',
	year TEXT DEFAULT 'Unknown'
)

{'id': UUID('4266db19-5d94-4bd5-bdfb-be33e3fcd03f'), 
'user_id': UUID('ad8a226d-6200-4ca6-9d3f-b3ad974f1c59'), 
'topic': None, 
'image_id': UUID('4266db19-5d94-4bd5-bdfb-be33e3fcd03f'), 
'created_at': datetime.datetime(2026, 2, 7, 2, 45, 4, 949370, tzinfo=datetime.timezone.utc), 
'local_route': 'User/4266db19-5d94-4bd5-bdfb-be33e3fcd03f.jpg', 
'name': 'Unknown', 
'artist': 'Unknown', 
'style': 'Unknown', 
'genre': 'Unknown', 
'year': 'Unknown', 
'owner_id': UUID('ad8a226d-6200-4ca6-9d3f-b3ad974f1c59')}