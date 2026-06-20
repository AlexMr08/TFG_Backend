from app.core.auth import create_access_token


if __name__ == "__main__":
    # Change the subject if you need a specific user id
    print(create_access_token("022a7dbe-ce1a-4d6b-9d31-3f8be40b3d8d"))
