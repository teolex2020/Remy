"""Push notification routes for the local web UI."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/push/vapid-key")
async def get_vapid_key():
    from remy.web.push import get_vapid_keys

    public_key, _ = get_vapid_keys()
    return {"public_key": public_key}


@router.post("/push/subscribe")
async def push_subscribe(request: Request):
    body = await request.json()
    from remy.web.push import save_subscription

    save_subscription(body)
    return {"status": "subscribed"}


@router.post("/push/unsubscribe")
async def push_unsubscribe():
    from remy.web.push import remove_subscription

    remove_subscription()
    return {"status": "unsubscribed"}


@router.get("/push/status")
async def push_status():
    from remy.web.push import get_subscription

    sub = get_subscription()
    return {"subscribed": sub is not None}
