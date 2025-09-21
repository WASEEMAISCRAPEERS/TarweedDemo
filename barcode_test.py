from pyzbar.pyzbar import decode
import cv2
import json
import textwrap  # for wrapping long text

# Load products database from JSON
with open("products.json", "r") as file:
    products_db = json.load(file)

# Load image (replace with your file path)
img = cv2.imread("barcode.png")

# Decode barcodes
barcodes = decode(img)

if barcodes:
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        print(f"✅ Found {barcode_type}: {barcode_data}")

        # Lookup product in JSON database
        product_info = products_db.get(barcode_data, "❌ Not found in database")
        print("📦 Product:", product_info)

        # Draw rectangle around barcode
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Combine barcode + product info
        overlay_text = f"{barcode_data} | {product_info}"

        # Wrap text so it doesn’t get cut off (40 characters per line)
        wrapped_text = textwrap.wrap(overlay_text, width=40)

        # Draw each line separately on the image
        y_offset = 50
        for line in wrapped_text:
            cv2.putText(img, line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30  # move down for next line

    # Show full image in a resizable window
    cv2.namedWindow("Barcode Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Barcode Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output image (so you can view full frame in any image viewer)
    cv2.imwrite("barcode_result.png", img)
    print("💾 Saved output as barcode_result.png")

else:
    print("❌ No barcode detected in the image.")
