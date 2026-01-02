const SCRAPED_URL =
  "public_view/scraped_publications_metadata.json";
const PREPROCESSED_URL =
  "public_view/preprocessed_publications_metadata.json";
const TKG_URL = "public_view/tkg_raw_metadata.json";

document.addEventListener("DOMContentLoaded", () => {
  loadAndRenderHeatmap().catch((err) => {
    console.error(err);
    const container = document.getElementById("heatmap-container");
    if (container) {
      container.textContent = "Failed to load publication metadata.";
    }
  });

  loadAndRenderTKG().catch((err) => {
    console.error(err);
    const container = document.getElementById("tkg-container");
    if (container) {
      container.textContent = "Failed to load TKG metadata.";
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

async function loadAndRenderTKG() {
  const tkgRes = await fetch(TKG_URL);

  if (!tkgRes.ok) {
    throw new Error("Error fetching TKG metadata JSON file");
  }

  const tkgData = await tkgRes.json();
  buildTKGVisualization(tkgData);
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

function buildTKGVisualization(tkgData) {
  const container = document.getElementById("tkg-container");
  if (!container) return;

  container.innerHTML = "";

  const publishers = tkgData.publishers || [];

  // Map publishers -> Map(dateKey -> [pubs])
  const byPublisherDate = new Map();
  const dateSet = new Set();

  publishers.forEach((pub) => byPublisherDate.set(pub, new Map()));

  (tkgData.publications || []).forEach((pub) => {
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
    dateMap.get(dateKey).push(pub);
  });

  const dates = Array.from(dateSet).sort();

  const table = document.createElement("table");
  table.className = "tkg-table";

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
    th.textContent = `${d}.${m}`;
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
      td.className = "tkg-cell";

      const pubsForDate = dateMap.get(dateKey);
      if (pubsForDate && pubsForDate.length > 0) {
        const pubContainer = document.createElement("div");
        pubContainer.className = "tkg-pub-container";

        pubsForDate.forEach((pub) => {
          const pubMatrix = createPublicationMatrix(pub);
          pubContainer.appendChild(pubMatrix);
        });

        td.appendChild(pubContainer);
      }

      row.appendChild(td);
    });

    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  container.appendChild(table);

  setTimeout(() => {
    container.scrollLeft = container.scrollWidth;
  }, 0);
}

function createPublicationMatrix(pub) {
  const matrix = document.createElement("div");
  matrix.className = "event-matrix";
  matrix.title = `${pub.title}\n${pub.n_events} events`;

  const events = pub.events || [];

  // Add labels column first (only once for all events)
  const labelsColumn = document.createElement("div");
  labelsColumn.className = "event-labels-column";

  const labels = [
    "Statement",
    "Temporal",
    "Temp.Conf",
    "Valid@",
    "Invalid@",
    "Invalidated",
    "Triplets",
    "Entities",
    "Resolved"
  ];

  labels.forEach(labelText => {
    const label = document.createElement("div");
    label.className = "box-label";
    label.textContent = labelText;
    labelsColumn.appendChild(label);
  });

  matrix.appendChild(labelsColumn);

  events.forEach((event) => {
    const eventColumn = document.createElement("div");
    eventColumn.className = "event-column";

    // Box 1: statement_type
    const statementBox = createBox(
      getStatementTypeConfig(event.statement_type),
      `Statement: ${event.statement_type}`
    );
    eventColumn.appendChild(statementBox);

    // Box 2: temporal_type
    const temporalBox = createBox(
      getTemporalTypeConfig(event.temporal_type),
      `Temporal: ${event.temporal_type}`
    );
    eventColumn.appendChild(temporalBox);

    // Box 3: temporal_confidence
    const tempConfBox = createBox(
      getConfidenceConfig(event.temporal_confidence),
      `Temporal Confidence: ${event.temporal_confidence}`
    );
    eventColumn.appendChild(tempConfBox);

    // Box 4: valid_at
    const validAtBox = createDateBox(
      event.valid_at,
      event.valid_at_confidence,
      "Valid At"
    );
    eventColumn.appendChild(validAtBox);

    // Box 5: invalid_at
    const invalidAtBox = createDateBox(
      event.invalid_at,
      event.invalid_at_confidence,
      "Invalid At"
    );
    eventColumn.appendChild(invalidAtBox);

    // Box 6: invalidated_by
    const invalidatedBox = createBox(
      {
        color: event.invalidated_by ? "#4caf50" : "#bdbdbd",
        label: event.invalidated_by ? "✓" : "∅",
      },
      `Invalidated By: ${event.invalidated_by || "None"}`
    );
    eventColumn.appendChild(invalidatedBox);

    // Box 7: n_triplets
    const tripletsBox = createCountBox(
      event.n_triplets,
      `Triplets: ${event.n_triplets}`
    );
    eventColumn.appendChild(tripletsBox);

    // Box 8: n_entities
    const entitiesBox = createCountBox(
      event.n_entities,
      `Entities: ${event.n_entities}`
    );
    eventColumn.appendChild(entitiesBox);

    // Box 9: n_resolved_entities
    const resolvedBox = createCountBox(
      event.n_resolved_entities,
      `Resolved: ${event.n_resolved_entities}`
    );
    eventColumn.appendChild(resolvedBox);

    matrix.appendChild(eventColumn);
  });

  // Make matrix clickable to open URL
  matrix.addEventListener("click", (e) => {
    e.preventDefault();
    if (pub.url) {
      window.open(pub.url, "_blank", "noopener,noreferrer");
    }
  });

  return matrix;
}

function createBox(config, title) {
  const box = document.createElement("div");
  box.className = "tkg-box";
  box.style.backgroundColor = config.color;
  box.textContent = config.label;
  box.title = title;
  return box;
}

function createDateBox(dateValue, confidence, label) {
  if (!dateValue) {
    return createBox({ color: "#bdbdbd", label: "∅" }, `${label}: Empty`);
  }

  const confConfig = getConfidenceConfig(confidence);
  const dateStr = dateValue.split("T")[0];

  return createBox(
    { color: confConfig.color, label: "📅" },
    `${label}: ${dateStr}\nConfidence: ${confidence}`
  );
}

function createCountBox(count, title) {
  const color = count > 0 ? "#9e9e9e" : "#ef5350";
  return createBox({ color, label: String(count) }, title);
}

function getStatementTypeConfig(type) {
  const configs = {
    FACT: { color: "#2196f3", label: "F" },
    OPINION: { color: "#ef5350", label: "O" },
    PREDICTION: { color: "#ff9800", label: "P" },
  };
  return configs[type] || { color: "#bdbdbd", label: "?" };
}

function getTemporalTypeConfig(type) {
  const configs = {
    ATEMPORAL: { color: "#ef5350", label: "A" },
    EVENT: { color: "#2196f3", label: "E" },
    STATE: { color: "#4caf50", label: "S" },
    FORECAST: { color: "#ff9800", label: "F" },
  };
  return configs[type] || { color: "#bdbdbd", label: "?" };
}

function getConfidenceConfig(confidence) {
  const configs = {
    LOW: { color: "#ef5350", label: "L" },
    MEDIUM: { color: "#ff9800", label: "M" },
    HIGH: { color: "#4caf50", label: "H" },
  };
  return configs[confidence] || { color: "#bdbdbd", label: "?" };
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