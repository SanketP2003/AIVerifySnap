"use client";

import { ScanText, Users, AlertTriangle, ShieldAlert } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

const chartData = [
    { name: 'Mon', scans: 120, fakes: 40 },
    { name: 'Tue', scans: 230, fakes: 90 },
    { name: 'Wed', scans: 340, fakes: 130 },
    { name: 'Thu', scans: 290, fakes: 110 },
    { name: 'Fri', scans: 450, fakes: 210 },
    { name: 'Sat', scans: 512, fakes: 280 },
    { name: 'Sun', scans: 600, fakes: 320 },
];

export default function AdminPage() {
    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 md:p-8 max-w-7xl mx-auto space-y-8 bg-background">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 border-b pb-6 gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tighter">System Overview</h1>
                    <p className="text-muted-foreground mt-2">Monitor global verification metrics and threat intelligence.</p>
                </div>
                <div className="px-4 py-2 bg-primary/10 text-primary font-semibold rounded-full text-sm flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" /> Live Analysis Server Online
                </div>
            </div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6"
            >
                <div className="bg-card p-6 border rounded-[1.5rem] shadow-sm flex flex-col justify-between h-40">
                    <div className="flex justify-between items-start">
                        <div className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">Total Scans</div>
                        <ScanText className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                        <div className="text-4xl font-bold font-mono">1.2M</div>
                        <div className="text-sm text-green-500 font-medium mt-1">+12% from last month</div>
                    </div>
                </div>

                <div className="bg-card p-6 border rounded-[1.5rem] shadow-sm flex flex-col justify-between h-40">
                    <div className="flex justify-between items-start">
                        <div className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">Fake Rate</div>
                        <AlertTriangle className="w-5 h-5 text-destructive" />
                    </div>
                    <div>
                        <div className="text-4xl font-bold font-mono text-destructive">48%</div>
                        <div className="text-sm text-destructive font-medium mt-1">+4% from last month</div>
                    </div>
                </div>

                <div className="bg-card p-6 border rounded-[1.5rem] shadow-sm flex flex-col justify-between h-40">
                    <div className="flex justify-between items-start">
                        <div className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">Registered Users</div>
                        <Users className="w-5 h-5 text-blue-500" />
                    </div>
                    <div>
                        <div className="text-4xl font-bold font-mono">4,821</div>
                        <div className="text-sm text-green-500 font-medium mt-1">+2% from last month</div>
                    </div>
                </div>

                <div className="bg-card p-6 border rounded-[1.5rem] shadow-sm flex flex-col justify-between h-40">
                    <div className="flex justify-between items-start">
                        <div className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">Takedowns Sent</div>
                        <ShieldAlert className="w-5 h-5 text-yellow-500" />
                    </div>
                    <div>
                        <div className="text-4xl font-bold font-mono">842</div>
                        <div className="text-sm text-green-500 font-medium mt-1">+15% from last month</div>
                    </div>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-card border rounded-[2rem] p-6 lg:p-8 shadow-sm flex flex-col"
            >
                <h2 className="text-xl font-bold tracking-tight mb-6">Daily Activity Intelligence</h2>
                <div className="h-[400px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorScans" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorFakes" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}`} />
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'hsl(var(--card))', borderRadius: '1rem', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))' }}
                                itemStyle={{ color: 'hsl(var(--foreground))' }}
                            />
                            <Area type="monotone" dataKey="scans" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorScans)" strokeWidth={2} />
                            <Area type="monotone" dataKey="fakes" stroke="hsl(var(--destructive))" fillOpacity={1} fill="url(#colorFakes)" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>
        </div>
    );
}
