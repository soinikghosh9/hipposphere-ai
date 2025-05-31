
import React from 'react';

interface SectionProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  titleClassName?: string;
  subtitleClassName?: string;
  id?: string;
}

export const Section: React.FC<SectionProps> = ({ 
  title, 
  subtitle, 
  children, 
  className = '', 
  titleClassName = 'text-3xl font-bold text-accent-teal mb-3',
  subtitleClassName = 'text-lg text-sky-blue mb-6',
  id
}) => {
  return (
    <section className={`py-8 md:py-12 ${className}`} id={id}>
      {title && <h2 className={`${titleClassName}`}>{title}</h2>}
      {subtitle && <p className={`${subtitleClassName}`}>{subtitle}</p>}
      {children}
    </section>
  );
};
    