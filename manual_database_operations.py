from src.database import PostsDatabase


def delete_publication():
    """Delete one publication from the database."""
    db = PostsDatabase(db_path="./database/scraped_posts.db")
    db.delete_publication(table_name="amprion", publication_id="21b4555bb5674825fec82391fd41657ee7b1b7cc5a5d1dfda3af54c64e51c56c")
    db.close()

    db = PostsDatabase(db_path="./database/preprocessed_posts.db")
    db.delete_publication(table_name="amprion", publication_id="21b4555bb5674825fec82391fd41657ee7b1b7cc5a5d1dfda3af54c64e51c56c")
    db.close()

delete_publication()