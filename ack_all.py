from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone
import csv

SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]
ACK_WINDOW = timedelta(days=3)

def build_service(sa_path):
    creds = service_account.Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)

def main():
    package_name = "<PACKAGE_NAME>"
    service = build_service("service_account.json")
    now = datetime.now(timezone.utc)
    
    request = service.orders().list(packageName=package_name, pageSize=100)
    all_orders = []
    
    while request is not None:
        resp = request.execute()
        orders = resp.get("orders", [])
        all_orders.extend(orders)
        request = service.orders().list_next(request, resp)
    
    print(f"Trovati {len(all_orders)} ordini")
    out_rows = [["orderId", "purchaseToken", "productId", "state", "acknowledged", "result"]]

    for ord_obj in all_orders:
        order_id = ord_obj["orderId"]
        state = ord_obj["state"]
        token = ord_obj.get("purchaseToken")
        product_id = ord_obj.get("lineItems", [{}])[0].get("productId")
        ack_state = ord_obj.get("acknowledgementState")
        
        if state != "PROCESSED" or not token:
            out_rows.append([order_id, token or "", product_id or "", state, ack_state, "SKIP"])
            continue
        
        if ack_state == "ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED":
            out_rows.append([order_id, token, product_id, state, ack_state, "OK già"])
            continue

        try:
            service.purchases().subscriptions().acknowledge(
                packageName=package_name,
                subscriptionId=product_id,
                token=token,
                body={}
            ).execute()
            out_rows.append([order_id, token, product_id, state, ack_state, "ACK fatto"])
        except Exception as e:
            out_rows.append([order_id, token, product_id, state, ack_state, f"ERRORE {e}"])
    
    with open("ack_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

if __name__ == "__main__":
    main()
