from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class CheckLoginRequest(BaseModel):
    google_token: str = Field(..., description="Firebase ID token to verify")

class CheckEmailRequest(BaseModel):
    email: EmailStr
    
class SignUpWithGoogleRequest(BaseModel):
    google_token: Optional[str] = None
    name: str = Field(..., description="Name of the user")
    email: Optional[str] = None
    password: Optional[str] = None