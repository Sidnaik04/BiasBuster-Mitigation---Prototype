from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.db.database import Base, engine
from app.api.upload import router as upload_router
from app.api.baseline import router as baseline_router
from app.api.mitigation import router as mitigation_router
from app.api.auto_mitigation import router as auto_mitigation_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(upload_router)
app.include_router(baseline_router)
app.include_router(mitigation_router)
app.include_router(auto_mitigation_router)
