
from pydantic import BaseModel, Field

# 请求
class Request(BaseModel):
    text: str = Field(...,min_length=1,max_length=20000,description="待分析的文本")
    top_k:int = Field(default=5, ge=1,le=20,description="返回条数1-20")
# 回复
class Response(BaseModel):
    summary: str
    score: float =Field(...,ge=0,le=1)

# 联系人信息
class Contact(BaseModel):
    name: str | None = Field(default=None, description="联系人姓名")
    phone: str | None = Field(default=None, description="联系人电话")

# 文件结构化
class TenderNotice(BaseModel):
    project_name: str | None = Field(default=None, description="项目名称,没有则为None")
    buyer: str | None = Field(default=None, description="个人或单位名称,没有则为None")
    budget_wan: float | None = Field(default=None, description="预算,万元为单位,没有则为None")
    deadline: str | None = Field(default=None, description="截止时间,没有则为None")
    qualification: list[str] = Field(
        default_factory=list,description="资格要求,逐字拆解为字符串,没有则为空数组"
    )
    contact: Contact = Field(default_factory=Contact,description="联系人信息")
    confidence: float = Field(default=0.0,ge=0,le=1,description="你对本次抽取准确度的自评,0-1之间")

    # 降级路径用的空对象
    @classmethod
    def empty(cls) -> "TenderNotice":
        return cls()
