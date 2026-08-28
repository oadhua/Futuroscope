from pydantic import BaseModel, Field
from typing import List, Optional


class ColumnProfileSchema(BaseModel):
    column_name: str = Field(..., description="Tên cột dữ liệu")
    column_type: str = Field(
        ..., description="Kiểu dữ liệu chuẩn hóa (Integer, Float, Date, String...)"
    )
    raw_data_type: str = Field(
        ..., description="Kiểu dữ liệu gốc từ database/data frame"
    )
    column_description: str = Field(..., description="Mô tả ý nghĩa của cột")
    column_owner: str = Field(..., description="Nguồn sở hữu dữ liệu")

    class Config:
        from_attributes = True


class InitialProfilingResponse(BaseModel):
    selected_version: str = Field(..., description="Phiên bản dữ liệu được chọn")
    dataset_description: str = Field(..., description="Mô tả tổng quan về tập dữ liệu")
    total_columns: int = Field(..., description="Tổng số cột trong dataset")
    total_rows: int = Field(..., description="Tổng số dòng dữ liệu")
    columns: List[ColumnProfileSchema]


class DataVersionOption(BaseModel):
    version_id: str
    version_label: str
    is_updated: bool = False
