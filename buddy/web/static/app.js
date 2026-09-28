/* Buddy dashboard.
 *
 * Vanilla JS, no build pipeline, structured so a React port is mechanical:
 * each view is a function returning { el, refresh, dispose }, the router owns
 * which one is mounted, and every fetch goes through `api`.
 *
 * The DOM is built with `h`, never with innerHTML, so a task title, a brief,
 * or a line of agent output cannot become markup.
 *
 * This is a viewer. It has no route that changes anything.
 */

"use strict";

/* ------------------------------------------------------------------ util */

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "style") el.setAttribute("style", value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path) {
  const response = await fetch(path, { headers: { accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {
      /* a non-JSON error body is still an error */
    }
    throw new Error(detail);
  }
  return response.json();
}

function socket(path) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${scheme}://${location.host}${path}`);
}

function ago(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  const hrs = Math.floor(m / 60);
  if (hrs < 24) return `${hrs}h ${String(m % 60).padStart(2, "0")}m`;
  return `${Math.floor(hrs / 24)}d ${hrs % 24}h`;
}

function when(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString();
}

/* Enough to place something in the day, short enough not to be elided. */
function shortWhen(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function clockTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function bytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

const LABELS = { waiting_input: "needs input" };
const label = (status) => LABELS[status] || (status || "").replace(/_/g, " ");

function pill(status) {
  return h("span", {
    class: `pill ${status || "idle"}`,
    style: `--status: var(--${status || "idle"})`,
    title: status || "idle",
    text: label(status),
  });
}

function priority(value) {
  if (value === null || value === undefined) return null;
  const names = { 1: "urgent", 2: "high", 3: "normal", 4: "low", 5: "whenever" };
  return h("span", {
    class: `prio p${value}`,
    title: `priority ${value} - ${names[value] || "?"}`,
    text: `P${value}`,
  });
}

function taskLink(id, className) {
  return h("a", { class: className || "row-id", href: `#/task/${encodeURIComponent(id)}`, text: id });
}

function empty(message) {
  return h("div", { class: "empty", text: message });
}

/* -------------------------------------------------------------- live age */

/* Ages tick locally between polls, so a card never looks frozen. The base
 * comes from the server and elapsed client time is added to it, which keeps
 * a skewed clock out of the arithmetic. */
class Ticker {
  constructor() {
    this.entries = [];
    this.timer = setInterval(() => this.paint(), 1000);
  }
  track(el, baseSeconds, prefix, suffix) {
    if (baseSeconds === null || baseSeconds === undefined) return;
    this.entries.push({ el, base: baseSeconds, at: Date.now(), prefix: prefix || "", suffix: suffix || "" });
  }
  reset() {
    this.entries = [];
  }
  paint() {
    const now = Date.now();
    for (const entry of this.entries) {
      entry.el.textContent =
        entry.prefix + ago(entry.base + (now - entry.at) / 1000) + entry.suffix;
    }
  }
  dispose() {
    clearInterval(this.timer);
    this.entries = [];
  }
}

/* ------------------------------------------------------------- dashboard */

function DashboardView() {
  const agents = h("div", { class: "agents" });
  const agentsCount = h("span", { class: "count" });
  const queue = h("div", {});
  const queueCount = h("span", { class: "count" });
  const recent = h("div", {});
  const ideas = h("div", {});
  const ideasTitle = h("h2", { text: "Brainstorming" });
  const ideasCount = h("span", { class: "count" });
  // Hidden unless there is a brainstorm or a draft left from one: the usual
  // dashboard has nothing to say about it and should not make room for it.
  const ideasSection = h(
    "section",
    { class: "section brainstorm", hidden: true },
    h("div", { class: "section-head" }, h("span", { class: "thinking-dot" }), ideasTitle, ideasCount),
    ideas
  );
  const ticker = new Ticker();

  const el = h(
    "div",
    {},
    ideasSection,
    h(
      "section",
      { class: "section" },
      h("div", { class: "section-head" }, h("h2", { text: "Agents" }), agentsCount),
      agents
    ),
    h(
      "section",
      { class: "section" },
      h("div", { class: "section-head" }, h("h2", { text: "Queue" }), queueCount),
      queue
    ),
    h(
      "section",
      { class: "section" },
      h("div", { class: "section-head" }, h("h2", { text: "Recent tasks" })),
      recent
    )
  );

  function agentCard(agent) {
    const status = agent.status;
    const card = h("div", { class: `agent ${status}`, style: `--status: var(--${status})` });
    card.append(
      h("div", { class: "agent-head" }, h("span", { class: "agent-name", title: agent.name, text: agent.name }), pill(status)),
      h("div", { class: "agent-title", text: agent.title || agent.task_id }),
      h(
        "div",
        { class: "agent-sub" },
        taskLink(agent.task_id, "row-id"),
        priority(agent.priority),
        agent.project && h("span", { class: "tag", text: agent.project }),
        agent.harness && h("span", { class: "tag", text: agent.harness }),
        agent.attempt > 1 && h("span", { class: "tag", text: `attempt ${agent.attempt}` })
      )
    );
    if (agent.branch) {
      card.append(
        h(
          "div",
          { class: "agent-sub" },
          h("span", { class: "tag branch", title: agent.branch, text: agent.branch })
        )
      );
    }
    if (agent.tail) card.append(h("div", { class: "agent-tail", text: agent.tail }));

    const age = h("span", { text: "-" });
    const quiet = h("span", { text: "-" });
    ticker.track(age, agent.age_seconds, "up ");
    ticker.track(quiet, agent.last_output_ago_seconds, "quiet ");
    card.append(h("div", { class: "agent-foot" }, age, agent.last_output_ago_seconds !== null && quiet));
    card.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      location.hash = `#/task/${encodeURIComponent(agent.task_id)}`;
    });
    return card;
  }

  function queueRow(task) {
    const badges = (task.depends_on || []).map((dep) =>
      h("a", {
        class: "dep",
        href: `#/task/${encodeURIComponent(dep)}`,
        title: task.waiting_for || `depends on ${dep}`,
        text: `waits for ${dep}`,
      })
    );
    return h(
      "div",
      { class: "row" },
      h("span", {
        class: "row-pos",
        text: task.queue_position === null ? "-" : `${task.queue_position + 1}`,
      }),
      taskLink(task.id),
      h(
        "div",
        { class: "row-main" },
        h("div", { class: "row-title", text: task.title }),
        h(
          "div",
          { class: "row-sub" },
          h("span", { class: "agent-tag", title: task.agent, text: task.agent }),
          h("span", { text: task.project }),
          h("span", { class: "tag", text: task.harness }),
          badges
        )
      ),
      h("div", { class: "row-right" }, priority(task.priority))
    );
  }

  function draftRow(draft) {
    const after = (draft.after || []).map((ref) =>
      ref.startsWith("t-")
        ? h("a", { class: "dep", href: `#/task/${encodeURIComponent(ref)}`, text: `after ${ref}` })
        : h("span", { class: "dep", text: `after ${ref}` })
    );
    return h(
      "div",
      { class: "row" },
      h("span", { class: "row-pos" }),
      h("span", { class: "row-id draft-id", text: draft.id }),
      h(
        "div",
        { class: "row-main" },
        h("div", { class: "row-title", text: draft.title }),
        h(
          "div",
          { class: "row-sub" },
          h("span", { text: draft.project }),
          draft.harness && h("span", { class: "tag", text: draft.harness }),
          after
        ),
        draft.goal && h("div", { class: "draft-goal", text: draft.goal })
      ),
      h("div", { class: "row-right" }, draft.priority ? priority(draft.priority) : null)
    );
  }

  function paintBrainstorm(storm) {
    const drafts = storm.drafts || [];
    ideasSection.hidden = !storm.active && drafts.length === 0;
    ideasSection.classList.toggle("paused", !storm.active);
    ideasTitle.textContent = storm.active ? "Brainstorming" : "Drafts, not started";
    ideasCount.textContent = drafts.length
      ? `${drafts.length} draft${drafts.length === 1 ? "" : "s"} · nothing starts until /go`
      : "nothing starts until /go";
    ideas.replaceChildren(
      drafts.length
        ? h("div", { class: "rows" }, ...drafts.map(draftRow))
        : empty("Nothing drafted yet. Ideas become drafts as you talk them through.")
    );
  }

  function recentRow(task) {
    return h(
      "div",
      { class: "row" },
      h("span", { class: "row-pos" }),
      taskLink(task.id),
      h(
        "div",
        { class: "row-main" },
        h("div", { class: "row-title", text: task.title }),
        h(
          "div",
          { class: "row-sub" },
          task.agent && h("span", { class: "agent-tag", title: task.agent, text: task.agent }),
          h("span", { text: task.project }),
          task.branch && h("span", { class: "tag branch", title: task.branch, text: task.branch })
        )
      ),
      h("div", { class: "row-right" }, priority(task.priority), pill(task.state))
    );
  }

  async function refresh() {
    const [agentData, queued, history, storm] = await Promise.all([
      api("/api/agents"),
      api("/api/tasks?state=queued&limit=100"),
      api("/api/tasks?limit=15"),
      api("/api/brainstorm"),
    ]);
    ticker.reset();
    paintBrainstorm(storm);
    const running = agentData.agents;
    const limit = agentData.max_concurrent;
    agentsCount.textContent = running.length
      ? `${running.length} running${limit ? ` of ${limit}` : ""}`
      : "";
    agents.replaceChildren(
      ...(running.length
        ? running.map(agentCard)
        : [empty("No agents running. Queued work starts here as soon as it can.")])
    );
    queueCount.textContent = queued.tasks.length ? `${queued.tasks.length} waiting` : "";
    queue.replaceChildren(
      queued.tasks.length
        ? h("div", { class: "rows" }, ...queued.tasks.map(queueRow))
        : empty("Nothing queued. Every task that exists is running or finished.")
    );
    recent.replaceChildren(
      history.tasks.length
        ? h("div", { class: "rows" }, ...history.tasks.map(recentRow))
        : empty("No tasks yet. Say something to Buddy, or run `buddy spawn`.")
    );
    ticker.paint();
  }

  return { el, refresh, dispose: () => ticker.dispose() };
}

/* ------------------------------------------------------------- task view */

function TaskView(taskId) {
  const head = h("div", {});
  const meta = h("dl", { class: "meta" });
  const attempts = h("div", { class: "attempts" });
  const termWrap = h("div", { class: "term-wrap" });
  const brief = h("pre", { class: "block scroll", text: "" });
  const diff = h("pre", { class: "block scroll", text: "" });
  const result = h("pre", { class: "block scroll", text: "" });
  const ticker = new Ticker();

  const el = h(
    "div",
    {},
    h("div", { class: "crumb" }, h("a", { href: "#/", text: "Dashboard" }), h("span", { text: "/" }), h("span", { text: taskId })),
    head,
    meta,
    h(
      "section",
      { class: "section" },
      h("div", { class: "section-head" }, h("h2", { text: "Output" }), attempts),
      termWrap
    ),
    h(
      "div",
      { class: "cols" },
      h(
        "section",
        { class: "section" },
        h("div", { class: "section-head" }, h("h2", { text: "Brief" })),
        brief
      ),
      h(
        "div",
        {},
        h(
          "section",
          { class: "section" },
          h("div", { class: "section-head" }, h("h2", { text: "Diff vs base" })),
          diff
        ),
        h(
          "section",
          { class: "section" },
          h("div", { class: "section-head" }, h("h2", { text: "Result" })),
          result
        )
      )
    )
  );

  let terminal = null;
  let fit = null;
  let logSocket = null;
  let observer = null;
  let shownAttempt = null;

  function openTerminal() {
    if (terminal || typeof window.Terminal !== "function") return;
    terminal = new window.Terminal({
      convertEol: false,
      disableStdin: true,
      cursorBlink: false,
      cursorStyle: "bar",
      scrollback: 20000,
      fontSize: 12,
      fontFamily: 'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace',
      theme: {
        background: "#0a0d12",
        foreground: "#dfe5ee",
        cursor: "#0a0d12",
        selectionBackground: "#2b3648",
      },
    });
    if (window.FitAddon) {
      fit = new window.FitAddon.FitAddon();
      terminal.loadAddon(fit);
    }
    terminal.open(termWrap);
    if (fit) {
      fit.fit();
      observer = new ResizeObserver(() => {
        try {
          fit.fit();
        } catch (_) {
          /* the pane can be measured mid-teardown */
        }
      });
      observer.observe(termWrap);
    }
  }

  function follow(attempt) {
    if (attempt === shownAttempt) return;
    shownAttempt = attempt;
    for (const button of attempts.children) {
      button.classList.toggle("active", Number(button.dataset.attempt) === attempt);
    }
    if (logSocket) {
      logSocket.close();
      logSocket = null;
    }
    openTerminal();
    if (!terminal) {
      termWrap.replaceChildren(empty("xterm.js did not load, so the log cannot be rendered here."));
      return;
    }
    terminal.reset();
    logSocket = socket(`/ws/logs/${encodeURIComponent(taskId)}/${attempt}`);
    logSocket.onmessage = (event) => terminal.write(event.data);
    logSocket.onclose = (event) => {
      if (event.code === 1008) terminal.write(`\r\n\x1b[31m${event.reason}\x1b[0m\r\n`);
    };
  }

  function metaCell(name, value, options) {
    const { title } = options || {};
    return h(
      "div",
      {},
      h("dt", { text: name }),
      h("dd", { text: value ?? "-", title: title || value || "" })
    );
  }

  async function refresh() {
    const data = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
    const task = data.task;
    ticker.reset();

    head.replaceChildren(
      h(
        "div",
        { class: "task-head" },
        h("span", { class: "task-id", text: task.id }),
        h("h1", { text: task.title }),
        pill(task.state),
        priority(task.priority)
      )
    );

    const latest = data.runs[data.runs.length - 1];
    meta.replaceChildren(
      metaCell("Project", task.project),
      metaCell("Harness", task.harness),
      metaCell("Branch", data.branch),
      metaCell("Agent", task.agent || "-"),
      metaCell("Attempts", String(data.runs.length || task.attempt)),
      metaCell("Created", shortWhen(task.created_at), { title: when(task.created_at) }),
      latest && metaCell("Exit code", latest.exit_code === null ? "-" : String(latest.exit_code)),
      latest && metaCell("Outcome", latest.outcome || "-")
    );
    if (task.waiting_for) {
      meta.append(metaCell("Waiting", task.waiting_for, { title: task.waiting_for }));
    }

    attempts.replaceChildren(
      ...data.runs.map((run) =>
        h("button", {
          type: "button",
          dataset: { attempt: String(run.attempt) },
          text: `attempt ${run.attempt} - ${bytes(run.log_bytes)}`,
          onclick: () => follow(run.attempt),
        })
      )
    );

    brief.textContent = task.brief || "(no brief recorded)";
    diff.textContent = data.diff_error ? `no diff available: ${data.diff_error}` : data.diff_stat || "(no changes)";
    diff.classList.toggle("dim", Boolean(data.diff_error) || !data.diff_stat);
    result.textContent = data.result
      ? JSON.stringify(data.result, null, 2)
      : "(no result.json yet - the attempt has not finished)";
    result.classList.toggle("dim", !data.result);

    if (data.runs.length) {
      const wanted = shownAttempt && data.runs.some((run) => run.attempt === shownAttempt)
        ? shownAttempt
        : data.runs[data.runs.length - 1].attempt;
      follow(wanted);
    } else {
      termWrap.replaceChildren(empty("This task has not run yet."));
    }
    ticker.paint();
  }

  function dispose() {
    ticker.dispose();
    if (logSocket) logSocket.close();
    if (observer) observer.disconnect();
    if (terminal) terminal.dispose();
  }

  return { el, refresh, dispose };
}

/* ---------------------------------------------------------- conversation */

function ConversationView() {
  const chat = h("div", { class: "chat" });
  const el = h(
    "div",
    { class: "chat-column" },
    h("div", { class: "section-head" }, h("h2", { text: "Conversation" })),
    chat
  );
  let oldest = null;
  let live = null;
  let seen = new Set();

  function bubble(turn) {
    return h(
      "div",
      { class: `turn ${turn.speaker}` },
      h("span", { text: turn.text }),
      h("span", { class: "turn-when", title: when(turn.ts), text: clockTime(turn.ts) })
    );
  }

  function append(turn) {
    if (turn.id !== null && seen.has(turn.id)) return;
    if (turn.id !== null) seen.add(turn.id);
    chat.append(bubble(turn));
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  async function loadOlder(button) {
    const older = await api(`/api/conversation?before=${oldest}&limit=50`);
    button.remove();
    if (!older.turns.length) return;
    oldest = older.turns[0].id;
    const nodes = older.turns.map((turn) => {
      seen.add(turn.id);
      return bubble(turn);
    });
    if (older.has_more) nodes.unshift(moreButton());
    chat.prepend(...nodes);
  }

  function moreButton() {
    const button = h("button", { class: "more", type: "button", text: "Load earlier turns" });
    button.addEventListener("click", () => loadOlder(button));
    return button;
  }

  async function refresh() {
    // The live socket keeps this view current; a full reload is only needed
    // when there is nothing yet, or when that socket has gone away.
    if (chat.children.length && live) return;
    const data = await api("/api/conversation?limit=50");
    seen = new Set(data.turns.map((turn) => turn.id));
    oldest = data.turns.length ? data.turns[0].id : null;
    const nodes = data.turns.map(bubble);
    if (data.has_more) nodes.unshift(moreButton());
    chat.replaceChildren(
      ...(nodes.length ? nodes : [empty("Nothing said yet. Start a session with `buddy`.")])
    );
    window.scrollTo({ top: document.body.scrollHeight });

    if (!live) {
      live = socket("/ws/conversation");
      live.onmessage = (event) => append(JSON.parse(event.data));
      live.onclose = () => {
        live = null;
      };
    }
  }

  return {
    el,
    refresh,
    dispose: () => {
      if (live) live.close();
    },
  };
}

/* ----------------------------------------------------------------- shell */

const view = document.getElementById("view");
const conn = document.getElementById("conn");
let current = null;
let pending = null;

function route() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("task/")) return { name: "task", id: decodeURIComponent(hash.slice(5)) };
  if (hash.startsWith("conversation")) return { name: "conversation" };
  return { name: "dashboard" };
}

function markTabs(name) {
  for (const tab of document.querySelectorAll("#tabs a")) {
    tab.classList.toggle("active", tab.dataset.route === name);
  }
}

async function mount() {
  const target = route();
  if (current) current.dispose();
  markTabs(target.name);
  current =
    target.name === "task"
      ? TaskView(target.id)
      : target.name === "conversation"
        ? ConversationView()
        : DashboardView();
  view.replaceChildren(current.el);
  await refresh();
}

async function refresh() {
  if (!current) return;
  try {
    await current.refresh();
  } catch (error) {
    view.replaceChildren(h("div", { class: "error-box", text: String(error.message || error) }));
  }
}

/* Events arrive over the socket, but the socket is a nicety: the poll below
 * means a dropped connection costs freshness, never correctness. */
function scheduleRefresh() {
  if (pending) return;
  pending = setTimeout(() => {
    pending = null;
    refresh();
  }, 150);
}

function connect() {
  const events = socket("/ws/events");
  events.onopen = () => {
    conn.className = "conn live";
    conn.querySelector(".conn-text").textContent = "live";
  };
  events.onmessage = scheduleRefresh;
  events.onclose = () => {
    conn.className = "conn lost";
    conn.querySelector(".conn-text").textContent = "reconnecting";
    setTimeout(connect, 2000);
  };
  events.onerror = () => events.close();
}

window.addEventListener("hashchange", mount);
setInterval(() => {
  if (!document.hidden) refresh();
}, 5000);
connect();
mount();
