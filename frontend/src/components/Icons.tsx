import type { SVGProps } from "react";

const base: SVGProps<SVGSVGElement> = {
  width: 22,
  height: 22,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export const IconHome = () => (
  <svg {...base}>
    <path d="M3 11.5 12 4l9 7.5" />
    <path d="M5 10v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9" />
  </svg>
);

export const IconRadar = () => (
  <svg {...base}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="4.5" />
    <path d="M12 12 19 6.5" />
  </svg>
);

export const IconBell = () => (
  <svg {...base}>
    <path d="M6 9a6 6 0 1 1 12 0c0 6 2.5 7.5 2.5 7.5h-17S6 15 6 9Z" />
    <path d="M10 20a2 2 0 0 0 4 0" />
  </svg>
);

export const IconGraph = () => (
  <svg {...base}>
    <circle cx="6" cy="7" r="2.3" />
    <circle cx="18" cy="6" r="2.3" />
    <circle cx="12" cy="18" r="2.3" />
    <path d="M8.2 7.3 15.8 6.5M7.2 9.1 10.9 15.9M16.9 8.1 13.1 15.9" />
  </svg>
);

export const IconSearch = () => (
  <svg {...base}>
    <circle cx="10.5" cy="10.5" r="6.5" />
    <path d="M20 20 15.2 15.2" />
  </svg>
);

export const IconFlow = () => (
  <svg {...base}>
    <circle cx="4.5" cy="6" r="2" />
    <circle cx="4.5" cy="18" r="2" />
    <circle cx="19.5" cy="12" r="2" />
    <path d="M6.3 6.8 17.7 11.3M6.3 17.2 17.7 12.7" />
  </svg>
);

export const IconUpload = () => (
  <svg {...base}>
    <path d="M12 15V4M7 8.5 12 4l5 4.5" />
    <path d="M4.5 15.5V18a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2.5" />
  </svg>
);

export const IconList = () => (
  <svg {...base}>
    <path d="M8.5 6h11M8.5 12h11M8.5 18h11" />
    <path d="M4 6h.01M4 12h.01M4 18h.01" strokeWidth="2.6" />
  </svg>
);

export const IconPlug = () => (
  <svg {...base}>
    <path d="M9 3v5M15 3v5" />
    <path d="M6.5 8h11v3a5.5 5.5 0 0 1-11 0V8Z" />
    <path d="M12 16.5V21" />
  </svg>
);

export const IconShield = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6L12 3Z" />
    <path d="m9 12 2.2 2.2L15.5 10" />
  </svg>
);
