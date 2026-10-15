import "./globals.css";

export const metadata = {
  title: "XAI-AMLBench | Compliance Console",
  description: "Explainable graph-based AML alert review",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}