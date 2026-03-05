from fastapi import APIRouter
from app.mitigation.strategy_ranker import find_best_strategy

router = APIRouter(prefix="/auto-mitigation", tags=["Auto Mitigation"])


@router.post("/rank")
def auto_rank_strategies(strategy_results: dict):

    result = find_best_strategy(strategy_results)

    return {
        "status": "success",
        "best_strategy": result["best_strategy"],
        "ranking": result["ranking"],
    }
