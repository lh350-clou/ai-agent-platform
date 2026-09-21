"""SQLAlchemy 声明式基类：所有数据库模型都继承这里的 Base。

本阶段（2a）只打通连接，还没有业务表，所以这里不定义任何具体字段。
按 CLAUDE.md「没有第二个调用方之前不要提前抽取基类」的约定，
也暂时不做 TimestampMixin 之类的公共字段抽象 —— 等真的出现第二张表再说。
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# 约束命名规范。
# 不指定时，索引、唯一约束、外键、主键会由 PostgreSQL 自动命名，名字不可预测。
# 下一阶段接入 Alembic 后，autogenerate 需要靠「名字」来识别约束 —— 比如要删掉一个
# 唯一约束，就得先知道它叫什么。提前定好规则，迁移脚本才能稳定生成。
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。

    DeclarativeBase 是 SQLAlchemy 2.x 的写法：
    类属性上的 Mapped[...] + mapped_column() 会直接被翻译成数据库的列定义。
    """

    # 让 Base.metadata 使用上面定义的命名规则
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
