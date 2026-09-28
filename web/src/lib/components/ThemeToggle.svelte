<script lang="ts">
  // Three states, one control: system → dark → light. Icons rather than
  // words — a word has to be read, a sun does not, and the header should
  // stay quiet.
  type Theme = "system" | "dark" | "light";
  let theme = $state<Theme>("system");

  const NEXT: Record<Theme, Theme> = { system: "dark", dark: "light", light: "system" };
  const LABEL: Record<Theme, string> = {
    system: "theme: follow system", dark: "theme: dark", light: "theme: light",
  };

  function cycle() {
    theme = NEXT[theme];
    const el = document.documentElement;
    if (theme === "system") el.removeAttribute("data-theme");
    else el.setAttribute("data-theme", theme);
    try { localStorage.setItem("iwashere.theme", theme); } catch { /* private window */ }
  }

  $effect(() => {
    let saved: Theme | null = null;
    try { saved = localStorage.getItem("iwashere.theme") as Theme | null; } catch { /* ignore */ }
    if (saved && saved !== "system") {
      theme = saved;
      document.documentElement.setAttribute("data-theme", saved);
    }
  });
</script>

<button class="round" onclick={cycle} title={LABEL[theme]} aria-label={LABEL[theme]}>
  {#if theme === "light"}
    <!-- sun -->
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="3.1" />
      <g class="rays">
        <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.1 3.1l1.4 1.4M11.5 11.5l1.4 1.4M12.9 3.1l-1.4 1.4M4.5 11.5l-1.4 1.4"/>
      </g>
    </svg>
  {:else if theme === "dark"}
    <!-- crescent moon -->
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M13.2 10.1A5.6 5.6 0 0 1 6 2.8a5.8 5.8 0 1 0 7.2 7.3z" class="fill" />
    </svg>
  {:else}
    <!-- half-filled circle: follow the system -->
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="5.4" />
      <path d="M8 2.6a5.4 5.4 0 0 1 0 10.8z" class="fill" />
    </svg>
  {/if}
</button>

<style>
  /* Sizing and the capsule skin come from whichever chrome hosts it; this
     only owns the icon. */
  button {
    display: grid; place-items: center; padding: 0;
    background: none; border: 0; border-radius: 999px;
    width: 26px; height: 26px;
    color: var(--muted); cursor: pointer;
    transition: color 140ms var(--ease);
  }
  button.round { width: auto; height: auto; }
  button:hover { color: var(--text); }
  svg { width: 15px; height: 15px; }
  svg :global(circle), svg :global(path) {
    fill: none; stroke: currentColor; stroke-width: 1.3;
    stroke-linecap: round; stroke-linejoin: round;
  }
  svg :global(.fill) { fill: currentColor; stroke: none; }
  @media (prefers-reduced-motion: reduce) { button { transition: none; } }
</style>
