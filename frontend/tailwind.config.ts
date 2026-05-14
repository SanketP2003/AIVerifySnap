import type { Config } from "tailwindcss";

const config: Config = {
	darkMode: ["class"],
	content: [
		"./src/pages/**/*.{ts,tsx}",
		"./src/components/**/*.{ts,tsx}",
		"./src/app/**/*.{ts,tsx}",
		"./src/lib/**/*.{ts,tsx}",
	],
	theme: {
		container: {
			center: true,
			padding: "2rem",
			screens: {
				"2xl": "1400px",
			},
		},
		extend: {
			fontFamily: {
				sans: ["Inter", "var(--font-sans)", "system-ui", "sans-serif"],
				serif: ["DM Serif Display", "var(--font-serif)", "Georgia", "serif"],
				mono: ["JetBrains Mono", "var(--font-geist-mono)", "monospace"],
			},
			fontSize: {
				"display-xl": ["5.5rem", { lineHeight: "1.05", letterSpacing: "-0.035em" }],
				"display-lg": ["4.5rem", { lineHeight: "1.08", letterSpacing: "-0.03em" }],
				"display": ["3.5rem", { lineHeight: "1.1", letterSpacing: "-0.025em" }],
				"heading-lg": ["2.25rem", { lineHeight: "1.15", letterSpacing: "-0.02em" }],
				"heading": ["1.75rem", { lineHeight: "1.2", letterSpacing: "-0.015em" }],
				"body-lg": ["1.125rem", { lineHeight: "1.65", letterSpacing: "-0.01em" }],
				"body": ["1rem", { lineHeight: "1.6", letterSpacing: "-0.01em" }],
				"caption": ["0.8125rem", { lineHeight: "1.5", letterSpacing: "0" }],
			},
			colors: {
				background: "hsl(var(--background))",
				foreground: "hsl(var(--foreground))",
				card: {
					DEFAULT: "hsl(var(--card))",
					foreground: "hsl(var(--card-foreground))",
				},
				popover: {
					DEFAULT: "hsl(var(--popover))",
					foreground: "hsl(var(--popover-foreground))",
				},
				primary: {
					DEFAULT: "hsl(var(--primary))",
					foreground: "hsl(var(--primary-foreground))",
				},
				secondary: {
					DEFAULT: "hsl(var(--secondary))",
					foreground: "hsl(var(--secondary-foreground))",
				},
				muted: {
					DEFAULT: "hsl(var(--muted))",
					foreground: "hsl(var(--muted-foreground))",
				},
				accent: {
					DEFAULT: "hsl(var(--accent))",
					foreground: "hsl(var(--accent-foreground))",
				},
				destructive: {
					DEFAULT: "hsl(var(--destructive))",
					foreground: "hsl(var(--destructive-foreground))",
				},
				border: "hsl(var(--border))",
				input: "hsl(var(--input))",
				ring: "hsl(var(--ring))",
				// Semantic status colors
				success: {
					DEFAULT: "#16a34a",
					light: "#dcfce7",
				},
				warning: {
					DEFAULT: "#d97706",
					light: "#fef9c3",
				},
			},
			borderRadius: {
				"2xl": "1.25rem",
				xl: "1rem",
				lg: "0.75rem",
				md: "0.5rem",
				sm: "0.375rem",
			},
			boxShadow: {
				"soft": "0 1px 3px hsl(var(--foreground) / 0.04), 0 4px 16px hsl(var(--foreground) / 0.03)",
				"medium": "0 2px 8px hsl(var(--foreground) / 0.05), 0 8px 24px hsl(var(--foreground) / 0.04)",
				"large": "0 4px 12px hsl(var(--foreground) / 0.06), 0 16px 48px hsl(var(--foreground) / 0.05)",
			},
			keyframes: {
				"accordion-down": {
					from: { height: "0" },
					to: { height: "var(--radix-accordion-content-height)" },
				},
				"accordion-up": {
					from: { height: "var(--radix-accordion-content-height)" },
					to: { height: "0" },
				},
				"fade-up": {
					"0%": { opacity: "0", transform: "translateY(16px)" },
					"100%": { opacity: "1", transform: "translateY(0)" },
				},
				"fade-in": {
					"0%": { opacity: "0" },
					"100%": { opacity: "1" },
				},
				"scale-in": {
					"0%": { opacity: "0", transform: "scale(0.96)" },
					"100%": { opacity: "1", transform: "scale(1)" },
				},
			},
			animation: {
				"accordion-down": "accordion-down 0.2s ease-out",
				"accordion-up": "accordion-up 0.2s ease-out",
				"fade-up": "fade-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards",
				"fade-in": "fade-in 0.5s ease-out forwards",
				"scale-in": "scale-in 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards",
			},
			transitionTimingFunction: {
				"smooth": "cubic-bezier(0.16, 1, 0.3, 1)",
				"spring": "cubic-bezier(0.34, 1.56, 0.64, 1)",
			},
		},
	},
	plugins: [require("tailwindcss-animate")],
};

export default config;
