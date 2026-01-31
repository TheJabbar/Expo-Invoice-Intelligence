from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from api.core.routes import router
from api.utils.scheduler.retrain_scheduler import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Invoice Intelligence API...")
    start_scheduler()
    yield
    # Shutdown (if needed)
    logger.info("Shutting down Invoice Intelligence API...")


app = FastAPI(
    title="Invoice Intelligence API",
    version="1.0",
    description="API for intelligent invoice processing with OCR and field extraction",
    lifespan=lifespan
)

# Include the routes
app.include_router(router)

logger.info("Invoice Intelligence API initialized")