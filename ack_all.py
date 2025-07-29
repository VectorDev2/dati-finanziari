from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone
import csv, os, json

SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]
ACK_WINDOW = timedelta(days=3)

def build_service_from_env():
    sa_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("Variabile d'ambiente SERVICE_ACCOUNT_JSON mancante.")
    info = json.loads(sa_json)  # ← niente file, carica direttamente il JSON
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)

def main():
    package_name = os.environ.get("PACKAGE_NAME")
    if not package_name:
        print("❌ Errore: variabile PACKAGE_NAME mancante.")
        return

    service = build_service_from_env()
    now = datetime.now(timezone.utc)
    cutoff_time = now - ACK_WINDOW

    print(f"🔎 Recupero ordini per app: {package_name}, ultimi 3 giorni...")

    request = service.orders().list(packageName=package_name, pageSize=100)
    all_orders = []
    while request is not None:
        resp = request.execute()
        orders = resp.get("orders", [])
        all_orders.extend(orders)
        request = service.orders().list_next(request, resp)

    print(f"📦 Trovati {len(all_orders)} ordini totali (filtro ultimi 3 giorni)...")

    out_rows = [["orderId", "purchaseToken", "productId", "state", "ack_state", "risultato"]]
    ack_count, skip_count, err_count = 0, 0, 0

    for ord_obj in all_orders:
        order_id = ord_obj.get("orderId")
        state = ord_obj.get("state")
        token = ord_obj.get("purchaseToken")
        create_time = ord_obj.get("createTime")
        ack_state = ord_obj.get("acknowledgementState")
        line_items = ord_obj.get("lineItems", [])
        product_id = line_items[0].get("productId") if line_items else None

        # filtro 3 giorni
        try:
            ct = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
            if ct < cutoff_time:
                out_rows.append([order_id, token or "", product_id or "", state, ack_state, "⏩ fuori finestra (vecchio)"])
                skip_count += 1
                continue
        except:
            pass

        if state != "PROCESSED" or not token:
            out_rows.append([order_id, token or "", product_id or "", state, ack_state, "⏩ non idoneo"])
            skip_count += 1
            continue

        if ack_state == "ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED":
            out_rows.append([order_id, token, product_id, state, ack_state, "✔ già acknowledged"])
            skip_count += 1
            continue

        try:
            service.purchases().subscriptions().acknowledge(
                packageName=package_name,
                subscriptionId=product_id,
                token=token,
                body={}
            ).execute()
            out_rows.append([order_id, token, product_id, state, ack_state, "✅ ACK eseguito"])
            ack_count += 1
        except Exception as e:
            out_rows.append([order_id, token, product_id, state, ack_state, f"❌ ERRORE: {e}"])
            err_count += 1

    with open("ack_report.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print(f"✅ Completato! Acknowledge: {ack_count} | Skip: {skip_count} | Errori: {err_count}")

if __name__ == "__main__":
    main()
