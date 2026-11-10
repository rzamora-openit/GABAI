## Conversation
id: int | None
messages: list[Message]
owner: User

## Message
id: int | None
role: str
content: str
created_at: datetime
conversation_id: int | None
owner: User

## Folder
id: int | None
name: str
created_at: datetime
conversations: list[Conversation]
owner: User

## User
id: int | None
type: UserType
studentId: str
email: str
name: str
username: str
password: str

## enum UserType
admin
student
