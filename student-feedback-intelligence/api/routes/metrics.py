from fastapi import APIRouter
from api.schemas.feedback import ModelMetricsResponse
from api.services.metrics_service import MetricsService

router = APIRouter(
    prefix="/api/v1/model-metrics",
    tags=["Model Performance"],
)

service = MetricsService.get_instance()


@router.get("/", response_model=ModelMetricsResponse)
def get_model_metrics():
    return service.get_metrics()
