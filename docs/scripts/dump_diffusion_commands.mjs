// Dumps the `sglang serve` commands the diffusion Cookbook builders render, as JSON on
// stdout, for scripts/ci/check_diffusion_cookbook_commands.py --generated.
//
//   node docs/scripts/dump_diffusion_commands.mjs > gen.json
//
// Each builder page renders through the real COMMAND ENGINE of _deployment.jsx: the
// starting selection for every supported hardware, every documented recipe, and each
// base/serve option on top of those. Selections the builder rejects (Copy disabled)
// are not emitted.
import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../..", import.meta.url));
const docs = join(root, "docs");

const engineSource = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
const start = engineSource.indexOf("// ==== COMMAND ENGINE ====");
const end = engineSource.indexOf("// ==== end COMMAND ENGINE ====");
if (start < 0 || end < start) throw new Error("_deployment.jsx is missing the COMMAND ENGINE markers");
const createEngine = new Function(
  "config",
  `${engineSource.slice(start, end)}
  return { renderCommand, findCell, resolveModelName, overlayDimSpecs, optionVisible,
    optionDisabled, visibleOptions, HARDWARE_CATALOG };`,
);

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
const RECIPE_META = new Set(["id", "default", "unverified", "label", "description"]);
// The builder's own head-address and node-rank inputs; the checker needs real values.
const LOCAL_PLACEHOLDERS = { NODE_RANK: "0", NODE0_IP: "127.0.0.1" };

const vendorOf = (engine, config, hw) => {
  const extra = (config.hardware || []).find((entry) => entry.id === hw);
  if (extra?.vendor) return extra.vendor;
  for (const [vendor, list] of Object.entries(engine.HARDWARE_CATALOG)) {
    if (list.some((entry) => entry.id === hw)) return vendor;
  }
  return null;
};

// Mirrors normalizeBuilderSelection and reseatHiddenPicks in _deployment.jsx, which sit
// outside the COMMAND ENGINE block.
const normalize = (engine, config, raw) => {
  const builder = config.commandBuilder;
  let out = { ...raw };
  for (const [key, bounds] of Object.entries(builder.resource?.limits || {})) {
    if (typeof bounds.disabledWhen === "function" && bounds.disabledWhen(out)) {
      out[key] = bounds.disabledValue ?? 0;
      continue;
    }
    const fallback = Number(builder.defaultSelection?.[key] ?? bounds.min ?? 1);
    const value = Number.parseInt(out[key], 10);
    const values = typeof bounds.allowedValues === "function"
      ? bounds.allowedValues(out)
      : (bounds.allowedValues || []);
    const bounded = Math.min(bounds.max, Math.max(bounds.min, Number.isFinite(value) ? value : fallback));
    out[key] = values.length
      ? values.reduce((closest, candidate) =>
          Math.abs(candidate - bounded) < Math.abs(closest - bounded) ? candidate : closest, values[0])
      : bounded;
  }
  for (const key of ["tp_size", "ulysses_degree", "ring_degree"]) {
    const value = Number.parseInt(out[key], 10);
    out[key] = Number.isFinite(value) && value > 0 ? value : 1;
  }
  out.topology_mode = out.topology_mode === "manual" ? "manual" : "auto";
  for (const spec of engine.overlayDimSpecs) {
    if (spec.kind === "number") continue;
    const options = engine.visibleOptions(spec, out).filter((option) => !engine.optionDisabled(option, out));
    if (options.length && !options.some((option) => option.id === out[spec.id])) {
      out = { ...out, [spec.id]: options[0].id };
    }
  }
  return out;
};

const recipePicks = (recipe) =>
  Object.fromEntries(Object.entries(recipe).filter(([key, value]) =>
    !RECIPE_META.has(key) && typeof value !== "function"));

// Seeds: the selection a reader lands on after picking each hardware, then each
// documented recipe as the builder's recipe chips apply it.
const seedSelections = (engine, config) => {
  const builder = config.commandBuilder;
  const base = {
    ...Object.fromEntries(engine.overlayDimSpecs.map((spec) =>
      [spec.id, spec.default ?? spec.options?.[0]?.id ?? ""])),
    ...builder.defaultSelection,
  };
  const recipes = builder.resource?.verifiedRecipes || [];
  const seeds = [];
  for (const hw of config.supportedHardware || []) {
    const recipe = recipes.find((entry) => entry.hw === hw && entry.default)
      || recipes.find((entry) => entry.hw === hw);
    seeds.push({ ...base, hw, ...(recipe ? recipePicks(recipe) : {}), topology_mode: "auto" });
  }
  for (const recipe of recipes) {
    seeds.push({ ...base, ...recipePicks(recipe), topology_mode: "manual" });
  }
  return seeds.map((seed) => normalize(engine, config, seed));
};

const builderSelections = (engine, config) => {
  const seeds = seedSelections(engine, config);
  const selections = [...seeds];
  for (const seed of seeds) {
    for (const spec of engine.overlayDimSpecs) {
      if (!["base", "serve"].includes(spec.scope || "base") || spec.kind === "number") continue;
      if (typeof spec.showWhen === "function" && !spec.showWhen(seed)) continue;
      for (const option of engine.visibleOptions(spec, seed)) {
        if (option.id === seed[spec.id] || engine.optionDisabled(option, seed)) continue;
        selections.push(normalize(engine, config, { ...seed, [spec.id]: option.id }));
      }
    }
  }
  return selections;
};

const cellSelections = (engine, config) => {
  const overlayDefaults = Object.fromEntries(engine.overlayDimSpecs.map((spec) =>
    [spec.id, spec.default ?? spec.options?.[0]?.id ?? ""]));
  return (config.cells || []).map((cell) => ({ ...cell.match, ...overlayDefaults }));
};

const dumpPage = async (page) => {
  const source = readFileSync(page, "utf8");
  if (!/<Deployment\b/.test(source)) return [];
  const path = source.match(/import\s+\{\s*config\s*\}\s+from\s+["']([^"']+)["']/)?.[1];
  if (!path) return [];
  const { config } = await import(
    "data:text/javascript," + encodeURIComponent(readFileSync(join(docs, path), "utf8")));
  const engine = createEngine(config);
  const placeholders = {
    ...Object.fromEntries(Object.entries(config.placeholders || {}).map(([key, spec]) =>
      [key, spec.default ?? ""])),
    ...LOCAL_PLACEHOLDERS,
  };
  const pagePath = relative(root, page).split("\\").join("/");
  const selections = config.commandBuilder
    ? builderSelections(engine, config)
    : cellSelections(engine, config);
  const entries = new Map();
  for (const sel of selections) {
    const cell = config.commandBuilder
      ? config.commandBuilder.resolveDeployment(sel)
      : engine.findCell(config.cells, sel);
    if (!cell || cell.redirect || (cell.builder?.errors || []).length) continue;
    const command = engine.renderCommand(cell, sel, placeholders, "python");
    const key = `${sel.hw}\n${command}`;
    if (entries.has(key)) continue;
    entries.set(key, {
      source: `${pagePath}#${new URLSearchParams(sel).toString()}`,
      hw: sel.hw,
      vendor: vendorOf(engine, config, sel.hw),
      model: engine.resolveModelName(sel),
      command,
    });
  }
  return [...entries.values()];
};

const pages = walk(join(docs, "cookbook/diffusion")).filter((path) => path.endsWith(".mdx")).sort();
const out = [];
for (const page of pages) out.push(...await dumpPage(page));
process.stdout.write(JSON.stringify(out, null, 1) + "\n");
console.error(`${out.length} commands from ${new Set(out.map((entry) => entry.source.split("#")[0])).size} pages`);
