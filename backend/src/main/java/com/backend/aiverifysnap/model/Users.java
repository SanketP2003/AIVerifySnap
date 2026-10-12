package com.backend.aiverifysnap.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
@Table(name = "users")
public class Users {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @Column(name = "name")
    private String name;

    @Column(name = "email")
    private String email;

    @Column(name = "password_hash")
    private String passwordHash;

    @Column(name = "role")
    private String role;

    @Column(name = "clerk_id", unique = true)
    private String clerkId;

    @Column(name = "plan")
    private String plan = "STARTER";

    @Column(name = "scans_today")
    private Integer scansToday = 0;

    @Column(name = "last_scan_date")
    private LocalDate lastScanDate;

    @Column(name = "api_key", unique = true)
    private String apiKey;

    @Column(name = "webhook_url")
    private String webhookUrl;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
}
