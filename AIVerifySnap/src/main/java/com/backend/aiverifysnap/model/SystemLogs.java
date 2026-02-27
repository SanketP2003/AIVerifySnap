package com.backend.aiverifysnap.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
@Table(name = "system_logs")
public class SystemLogs {

    @Id
    @Column(name = "log_id")
    private Long logId;

    @Column(name = "action_type")
    private String actionType;

    @Column(name = "details")
    private String details;

    @Column(name = "timestamp")
    private LocalDateTime timestamp;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private Users id;

}
