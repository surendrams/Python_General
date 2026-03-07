from dataclasses import dataclass, field

@dataclass
class Template:
    application : str = field(init=False, default="MyApplication")

@dataclass
class SQLTemplate (Template):
    dbType: str = field(default="Oracle", init=False)
    userName: str
    passWord: str
    upperName: str = field(init=False)

    def __post_init__(self):
        self.upperName = self.userName.upper()


@dataclass
class MongoTemplate (Template):
    dbType: str = field(default="MongoDB", init=False)
    userName: str
    passWord: str
    upperName: str = field(init=False)

    def __post_init__(self):
        self.upperName = self.userName.upper()