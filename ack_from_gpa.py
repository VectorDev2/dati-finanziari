from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta, timezone
import os, json, csv

SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]
ACK_WINDOW = timedelta(days=3)  # 72 ore

def build_service_from_env():
    sa_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("Variabile d'ambiente SERVICE_ACCOUNT_JSON mancante.")
    info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)

def parse_order_ids(s: str):
    if not s:
        return []
    # accetta separatori: virgola, spazio, newline
    raw = [x.strip() for x in s.replace(",", "\n").splitlines()]
    return [x for x in raw if x.startswith("GPA.")]

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    package_name = os.environ.get("PACKAGE_NAME")
    order_ids_str = os.environ.get("ORDER_IDS", "")
    if not package_name:
        print("❌ PACKAGE_NAME mancante")
        return
    order_ids = parse_order_ids(order_ids_str)
    if not order_ids:
        print("❌ Non ho ricevuto Order ID (GPA...). Fornisci ORDER_IDS nel workflow.")
        return

    service = build_service_from_env()
    now = datetime.now(timezone.utc)
    cutoff_time = now - ACK_WINDOW

    out_rows = [["orderId", "purchaseToken", "productId", "state", "ack_state", "createTime", "risultato"]]
    ack_ok = skip = err = 0

    for batch in chunked(order_ids, 1000):
        try:
            resp = service.orders().batchget(
                packageName=package_name,
                orderIds=batch
            ).execute()
            orders = resp.get("orders", [])
        except HttpError as e:
            for gpa in batch:
                out_rows.append([gpa, "", "", "", "", "", f"❌ ERRORE batchget: {e}"])
                err += 1
            continue

        for ord_obj in orders:
            gpa = ord_obj.get("orderId")
            state = ord_obj.get("state")
            token = ord_obj.get("purchaseToken")
            create_time = ord_obj.get("createTime")  # ISO 8601
            ack_state = ord_obj.get("acknowledgementState")
            line_items = ord_obj.get("lineItems", [])
            product_id = line_items[0].get("productId") if line_items else None

            # Filtro temporale: ultimi 3 giorni
            within = True
            if create_time:
                try:
                    ct = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                    within = (ct >= cutoff_time)
                except:
                    pass
            if not within:
                out_rows.append([gpa, token or "", product_id or "", state or "", ack_state or "", create_time or "", "⏩ fuori finestra (vecchio)"])
                skip += 1
                continue

            if state != "PROCESSED" or not token or not product_id:
                out_rows.append([gpa, token or "", product_id or "", state or "", ack_state or "", create_time or "", "⏩ non idoneo"])
                skip += 1
                continue

            if ack_state == "ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED":
                out_rows.append([gpa, token, product_id, state, ack_state, create_time or "", "✔ già acknowledged"])
                skip += 1
                continue

            # Prova acknowledge
            try:
                service.purchases().subscriptions().acknowledge(
                    packageName=package_name,
                    subscriptionId=product_id,
                    token=token,
                    body={}
                ).execute()
                out_rows.append([gpa, token, product_id, state, ack_state, create_time or "", "✅ ACK eseguito"])
                ack_ok += 1
            except HttpError as e:
                out_rows.append([gpa, token, product_id, state, ack_state, create_time or "", f"❌ ERRORE ack: {e}"])
                err += 1

    with open("ack_report.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print(f"✅ Completato. ACK={ack_ok}  SKIP={skip}  ERRORI={err}")
    print("Report: ack_report.csv")

if __name__ == "__main__":
    main()
