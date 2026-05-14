"use client";

import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import {
    ChevronDown, Menu, X, Sun, Moon,
    ScanLine, Eye, Upload, Layers,
    BookOpen, Code2, HelpCircle, FileText,
    Users, Mail, CreditCard, Zap,
    LayoutDashboard, History, ShieldAlert
} from "lucide-react";
import { useAuth, UserButton, SignInButton } from "@clerk/nextjs";
import { useTheme } from "next-themes";
import { motion, AnimatePresence } from "framer-motion";

interface DropdownItem {
    icon: React.ElementType;
    label: string;
    desc: string;
    href: string;
}

const platformItems: DropdownItem[] = [
    { icon: ScanLine, label: "Image Detection", desc: "Analyze single images for deepfakes", href: "/detect" },
    { icon: Upload, label: "Bulk Scan", desc: "Upload multiple images at once", href: "/detect?mode=bulk" },
    { icon: Eye, label: "Identity Protection", desc: "Scan the web for unauthorized likeness", href: "/protection" },
    { icon: History, label: "Scan History", desc: "View all past analysis reports", href: "/history" },
    { icon: LayoutDashboard, label: "Admin Dashboard", desc: "Analytics & system overview", href: "/admin" },
    { icon: Layers, label: "API Access", desc: "Integrate detection into your apps", href: "/docs" },
];

const resourceItems: DropdownItem[] = [
    { icon: BookOpen, label: "How It Works", desc: "Understand ELA & forensic analysis", href: "/#how-it-works" },
    { icon: Code2, label: "API Documentation", desc: "Developer reference and guides", href: "/docs" },
    { icon: FileText, label: "Case Studies", desc: "Real-world detection reports", href: "/history" },
    { icon: HelpCircle, label: "Help Center", desc: "FAQs and support resources", href: "/#faq" },
];

const companyItems: DropdownItem[] = [
    { icon: Users, label: "About Us", desc: "Our mission to fight deepfakes", href: "/#about" },
    { icon: CreditCard, label: "Pricing", desc: "Plans for individuals & teams", href: "/pricing" },
    { icon: LayoutDashboard, label: "Dashboard", desc: "Admin analytics & system overview", href: "/admin" },
    { icon: Mail, label: "Contact", desc: "Get in touch with our team", href: "/#contact" },
];

function MegaDropdown({ items, isOpen }: { items: DropdownItem[]; isOpen: boolean }) {
    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
                    className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-[380px] rounded-xl overflow-hidden z-50"
                    style={{
                        background: "hsl(var(--card) / 0.92)",
                        backdropFilter: "blur(20px) saturate(1.3)",
                        WebkitBackdropFilter: "blur(20px) saturate(1.3)",
                        border: "1px solid hsl(var(--border) / 0.5)",
                        boxShadow: "0 8px 32px hsl(var(--foreground) / 0.08), 0 1px 3px hsl(var(--foreground) / 0.04)",
                    }}
                >
                    <div className="p-1.5">
                        {items.map((item) => (
                            <Link
                                key={item.label}
                                href={item.href}
                                className="flex items-start gap-3 px-3.5 py-3 rounded-lg hover:bg-muted/60 transition-colors duration-150 group"
                            >
                                <div className="mt-0.5 p-1.5 rounded-md bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-colors duration-200">
                                    <item.icon className="w-4 h-4" strokeWidth={1.8} />
                                </div>
                                <div className="min-w-0">
                                    <div className="text-[14px] font-semibold text-foreground leading-tight tracking-tight">
                                        {item.label}
                                    </div>
                                    <div className="text-[12px] text-muted-foreground mt-0.5 leading-snug">
                                        {item.desc}
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}

function ThemeToggle() {
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);
    if (!mounted) return <div className="w-9 h-9" suppressHydrationWarning />;

    return (
        <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-all duration-200"
            aria-label="Toggle theme"
        >
            <AnimatePresence mode="wait" initial={false}>
                {theme === "dark" ? (
                    <motion.div key="sun" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.15 }}>
                        <Sun className="w-[18px] h-[18px]" strokeWidth={1.8} />
                    </motion.div>
                ) : (
                    <motion.div key="moon" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.15 }}>
                        <Moon className="w-[18px] h-[18px]" strokeWidth={1.8} />
                    </motion.div>
                )}
            </AnimatePresence>
        </button>
    );
}

function AuthSection() {
    const { isSignedIn, isLoaded } = useAuth();
    const [mounted, setMounted] = useState(false);

    useEffect(() => setMounted(true), []);

    // Prevent hydration mismatch by returning a placeholder of roughly the same size during SSR/hydration
    if (!mounted || !isLoaded) {
        return <div className="flex items-center gap-2 min-w-[140px] h-9" suppressHydrationWarning />;
    }

    return (
        <>
            {!isSignedIn ? (
                <>
                    <SignInButton mode="modal">
                        <button className="px-4 py-2 text-[14px] font-medium text-muted-foreground hover:text-foreground transition-colors">
                            Sign In
                        </button>
                    </SignInButton>
                    <Link href="/sign-up" className="btn-primary text-[14px] px-5 py-2">
                        Get Started
                    </Link>
                </>
            ) : (
                <>
                    <Link href="/detect" className="btn-primary text-[14px] px-5 py-2 gap-1.5">
                        <Zap className="w-3.5 h-3.5" /> Detect
                    </Link>
                    <UserButton appearance={{ elements: { avatarBox: "w-8 h-8 rounded-full ring-2 ring-border" } }} />
                </>
            )}
        </>
    );
}

export function HeaderNav() {
    const { isSignedIn, isLoaded } = useAuth();
    const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
    const [mobileOpen, setMobileOpen] = useState(false);
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);
    const navRef = useRef<HTMLElement>(null);

    const handleMouseEnter = (menu: string) => {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setActiveDropdown(menu);
    };
    const handleMouseLeave = () => {
        timeoutRef.current = setTimeout(() => setActiveDropdown(null), 150);
    };

    useEffect(() => {
        const handleClick = (e: MouseEvent) => {
            if (navRef.current && !navRef.current.contains(e.target as Node)) setActiveDropdown(null);
        };
        document.addEventListener("click", handleClick);
        return () => document.removeEventListener("click", handleClick);
    }, []);

    const navLinks = [
        { label: "Platform", items: platformItems, key: "platform" },
        { label: "Resources", items: resourceItems, key: "resources" },
        { label: "Company", items: companyItems, key: "company" },
    ];

    return (
        <>
            <header
                className="sticky top-0 z-40 w-full border-b"
                style={{
                    background: "hsl(var(--background) / 0.8)",
                    backdropFilter: "blur(16px) saturate(1.2)",
                    WebkitBackdropFilter: "blur(16px) saturate(1.2)",
                    borderColor: "hsl(var(--border) / 0.35)",
                }}
            >
                <div className="mx-auto max-w-[1280px] px-6">
                    <div className="flex h-16 items-center justify-between">
                        {/* Logo */}
                        <Link href="/" className="flex items-center transition-opacity hover:opacity-70">
                            <span className="text-[22px] font-bold tracking-tighter text-foreground lowercase">
                                ai<span className="font-light text-accent"><b>verify</b></span>snap
                            </span>
                        </Link>

                        {/* Desktop Nav */}
                        <nav ref={navRef} className="hidden lg:flex items-center gap-0.5">
                            {navLinks.map((link) => (
                                <div
                                    key={link.key}
                                    className="relative"
                                    onMouseEnter={() => handleMouseEnter(link.key)}
                                    onMouseLeave={handleMouseLeave}
                                >
                                    <button
                                        className={`flex items-center gap-1 px-3.5 py-2 text-[13px] font-medium tracking-wide uppercase transition-colors duration-150 rounded-md ${activeDropdown === link.key
                                                ? "text-foreground"
                                                : "text-muted-foreground hover:text-foreground"
                                            }`}
                                    >
                                        {link.label}
                                        <ChevronDown
                                            className={`w-3 h-3 transition-transform duration-200 ${activeDropdown === link.key ? "rotate-180" : ""}`}
                                            strokeWidth={2}
                                        />
                                    </button>
                                    <MegaDropdown items={link.items} isOpen={activeDropdown === link.key} />
                                </div>
                            ))}
                            <Link
                                href="/pricing"
                                className="px-3.5 py-2 text-[13px] font-medium tracking-wide uppercase text-muted-foreground hover:text-foreground transition-colors duration-150 rounded-md"
                            >
                                Pricing
                            </Link>
                        </nav>

                        {/* Right side */}
                        <div className="hidden lg:flex items-center gap-2">
                            <ThemeToggle />
                            <AuthSection />
                        </div>

                        {/* Mobile */}
                        <div className="flex lg:hidden items-center gap-2">
                            <ThemeToggle />
                            <button onClick={() => setMobileOpen(true)} className="p-2 text-foreground" aria-label="Open menu">
                                <Menu className="w-5 h-5" strokeWidth={1.8} />
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Mobile Drawer */}
            <AnimatePresence>
                {mobileOpen && (
                    <>
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50" onClick={() => setMobileOpen(false)} />
                        <motion.div
                            initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
                            transition={{ type: "spring", damping: 30, stiffness: 300 }}
                            className="fixed top-0 right-0 bottom-0 w-[300px] bg-card border-l border-border z-50 overflow-y-auto"
                        >
                            <div className="p-5">
                                <div className="flex items-center justify-between mb-6">
                                    <span className="text-lg font-bold tracking-tighter lowercase text-foreground">
                                        ai<span className="font-light text-accent">verify</span>snap
                                    </span>
                                    <button onClick={() => setMobileOpen(false)} className="p-1.5 rounded-md hover:bg-muted text-muted-foreground" aria-label="Close menu">
                                        <X className="w-5 h-5" strokeWidth={1.8} />
                                    </button>
                                </div>

                                {isLoaded && !isSignedIn && (
                                    <div className="flex flex-col gap-2.5 mb-6">
                                        <SignInButton mode="modal">
                                            <button className="w-full py-2.5 text-[14px] font-medium rounded-lg bg-muted text-foreground border border-border/50">Sign In</button>
                                        </SignInButton>
                                        <Link href="/sign-up" onClick={() => setMobileOpen(false)} className="w-full py-2.5 text-[14px] font-medium rounded-lg bg-foreground text-background text-center">
                                            Get Started
                                        </Link>
                                    </div>
                                )}
                                {isLoaded && isSignedIn && (
                                    <div className="flex items-center gap-3 mb-6 p-3 bg-muted/30 rounded-lg border border-border/50">
                                        <UserButton appearance={{ elements: { avatarBox: "w-8 h-8 rounded-full ring-2 ring-border" } }} />
                                        <span className="text-[13px] font-medium text-muted-foreground">Your Account</span>
                                    </div>
                                )}

                                {navLinks.map((group) => (
                                    <div key={group.key} className="mb-5">
                                        <h3 className="text-[11px] font-semibold tracking-[0.15em] uppercase text-muted-foreground mb-2.5 px-1">{group.label}</h3>
                                        <div className="space-y-0.5">
                                            {group.items.map((item) => (
                                                <Link key={item.label} href={item.href} onClick={() => setMobileOpen(false)}
                                                    className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-muted/50 transition-colors group">
                                                    <div className="p-1.5 rounded-md bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-colors">
                                                        <item.icon className="w-4 h-4" strokeWidth={1.8} />
                                                    </div>
                                                    <div>
                                                        <div className="text-[13px] font-semibold text-foreground">{item.label}</div>
                                                        <div className="text-[11px] text-muted-foreground leading-tight">{item.desc}</div>
                                                    </div>
                                                </Link>
                                            ))}
                                        </div>
                                    </div>
                                ))}

                                <div className="border-t border-border/40 pt-3">
                                    <Link href="/pricing" onClick={() => setMobileOpen(false)}
                                        className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-muted/50 transition-colors">
                                        <div className="p-1.5 rounded-md bg-accent/10 text-accent"><CreditCard className="w-4 h-4" strokeWidth={1.8} /></div>
                                        <span className="text-[13px] font-semibold text-foreground">Pricing</span>
                                    </Link>
                                </div>
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </>
    );
}
