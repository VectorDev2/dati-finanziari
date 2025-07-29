from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os, json

SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]

def build_service_from_env():
    sa_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("❌ Variabile d'ambiente SERVICE_ACCOUNT_JSON mancante.")
    info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)

def parse_order_ids(s: str):
    if not s:
        return []
    raw = [x.strip() for x in s.replace(",", "\n").splitlines()]
    return [x for x in raw if x.startswith("GPA.")]

def main():
    package_name = os.environ.get("PACKAGE_NAME")
    order_ids_str = os.environ.get("ORDER_IDS", "")

    if not package_name:
        print("❌ PACKAGE_NAME mancante")
        return

    order_ids = parse_order_ids(order_ids_str)
    if not order_ids:
        print("❌ Nessun Order ID valido trovato")
        return

    service = build_service_from_env()
    ack_ok = skip = err = 0

    print(f"🔎 Trovati {len(order_ids)} Order ID da processare\n")

    for gpa in order_ids:
        try:
            # Recupero i dettagli dell'ordine
            ord_obj = service.orders().get(
                packageName=package_name,
                orderId=gpa
            ).execute()

            token = ord_obj.get("purchaseToken")
            state = ord_obj.get("state")
            ack_state = ord_obj.get("acknowledgementState")
            product_id = ord_obj.get("lineItems", [{}])[0].get("productId")

            if state != "PROCESSED" or not token:
                print(f"⏩ {gpa}: NON idoneo (state={state})")
                skip += 1
                continue

            if ack_state == "ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED":
                print(f"✔ {gpa}: già acknowledged")
                skip += 1
                continue

            # ACKNOWLEDGE
            service.purchases().subscriptions().acknowledge(
                packageName=package_name,
                subscriptionId=product_id,
                token=token,
                body={}
            ).execute()
            print(f"✅ {gpa}: ACK eseguito con successo")
            ack_ok += 1

        except HttpError as e:
            print(f"❌ {gpa}: ERRORE {e}")
            err += 1

    print(f"\n--- RISULTATO ---\nACK={ack_ok}  SKIP={skip}  ERRORI={err}")

if __name__ == "__main__":
    main()
