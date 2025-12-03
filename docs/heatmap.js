const SCRAPED_URL =
  "public_view/scraped_publications_metadata.json";
const PREPROCESSED_URL =
  "public_view/preprocessed_publications_metadata.json";

document.addEventListener("DOMContentLoaded", () => {
  loadAndRenderHeatmap().catch((err) => {
    console.error(err);
    const container = document.getElementById("heatmap-container");
    if (container) {
      container.textContent = "Failed to load publication metadata.";
    }
  });
});

async function loadAndRenderHeatmap() {
  const [scrapedRes, preprocessedRes] = await Promise.all([
    fetch(SCRAPED_URL),
    fetch(PREPROCESSED_URL),
  ]);

  if (!scrapedRes.ok || !preprocessedRes.ok) {
    throw new Error("Error fetching metadata JSON files");
  }

  const scrapedData = await scrapedRes.json();
  const preprocessedData = await preprocessedRes.json();

  buildHeatmap(scrapedData, preprocessedData);
}

function buildHeatmap(scrapedData, preprocessedData) {
  const container = document.getElementById("heatmap-container");
  if (!container) return;

  container.innerHTML = "";

  const preprocessedIds = new Set(
    (preprocessedData.publications || []).map((p) => p.id)
  );

  const publishers = scrapedData.publishers || [];

  // Map publishers -> Map(dateKey -> [pubs])
  const byPublisherDate = new Map();
  const dateSet = new Set();

  publishers.forEach((pub) => byPublisherDate.set(pub, new Map()));

  (scrapedData.publications || []).forEach((pub) => {
    const publisher = pub.publisher;
    const dateKey = toDateKey(pub.published_on);
    if (!publisher || !dateKey) return;

    dateSet.add(dateKey);

    if (!byPublisherDate.has(publisher)) {
      byPublisherDate.set(publisher, new Map());
    }
    const dateMap = byPublisherDate.get(publisher);
    if (!dateMap.has(dateKey)) {
      dateMap.set(dateKey, []);
    }
    dateMap.get(dateKey).push({
      ...pub,
      inPreprocessed: preprocessedIds.has(pub.id),
    });
  });

  const dates = Array.from(dateSet).sort(); // ascending (oldest -> newest)

  const table = document.createElement("table");
  table.className = "heatmap-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");

  const cornerTh = document.createElement("th");
  cornerTh.className = "header-cell publisher-cell";
  cornerTh.textContent = "Publisher / Date";
  headerRow.appendChild(cornerTh);

  dates.forEach((dateKey) => {
    const th = document.createElement("th");
    th.className = "date-header";
    const [y, m, d] = dateKey.split("-");
    th.textContent = `${d}.${m}`; // DD.MM
    th.title = dateKey;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  publishers.forEach((publisher) => {
    const row = document.createElement("tr");

    const nameCell = document.createElement("th");
    nameCell.className = "publisher-cell";
    nameCell.textContent = publisher;
    row.appendChild(nameCell);

    const dateMap = byPublisherDate.get(publisher) || new Map();

    dates.forEach((dateKey) => {
      const td = document.createElement("td");
      td.className = "heatmap-cell";

      const pubsForDate = dateMap.get(dateKey);
      if (pubsForDate && pubsForDate.length > 0) {
        const stack = document.createElement("div");
        stack.className = "square-stack";

        pubsForDate.forEach((pub) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "square-button";

          const square = document.createElement("span");
          square.className =
            "square " +
            (pub.inPreprocessed ? "square-both" : "square-scraped-only");

          square.title = `${publisher}\n${pub.title}\n${pub.published_on}`;
          btn.appendChild(square);

          btn.addEventListener("click", (e) => {
            e.preventDefault();
            if (pub.url) {
              window.open(pub.url, "_blank", "noopener,noreferrer");
            }
          });

          stack.appendChild(btn);
        });

        td.appendChild(stack);
      }

      row.appendChild(td);
    });

    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  container.appendChild(table);

  // Scroll horizontally to show the latest dates (rightmost columns)
  // Use a small timeout to ensure layout is complete.
  setTimeout(() => {
    container.scrollLeft = container.scrollWidth;
  }, 0);
}

function toDateKey(isoString) {
  if (!isoString) return null;
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return null;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
